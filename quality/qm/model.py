"""Quality-module networks (section 2b).

build_model returns a network that outputs 9 raw scores; the prediction is sigmoid(score).
- densenet121: baseline.
- mcf_densenet121: MCF-style (C1; Fu et al., MICCAI 2019): DenseNet121 branches on RGB / HSV / LAB, feature-level
                     and prediction-level fusion. In training mode it returns [B, 5, 9] (3 branches, feature fusion,
                     final) for the weighted loss (losses.MultiOutputLoss); in eval mode only the final scores [B, 9].
- swin_t: Swin-T (C2), torchvision weights, input 512 x 672 used as is.
Every model sets `head_name` (the attribute holding all head parameters) so the optimiser can give the head its own
learning rate. Memory saving (recomputation in the backward pass) lives in forward, so parameter names never change.
"""
import math

import torch
import torchvision
from torch import nn
from torch.utils.checkpoint import checkpoint

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
MCF_LOSS_WEIGHTS = (0.1, 0.1, 0.1, 0.1, 0.6)  # official MCF-Net code, utils/trainer.py


def _logit(p):
    return math.log(p / (1 - p))


def build_model(name, n_out, pretrained=True, dropout=0.2, memory_efficient=False, label_mean=None):
    if name == "densenet121":
        weights = torchvision.models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        m = torchvision.models.densenet121(weights=weights, memory_efficient=memory_efficient)
        # global average pooling is inside torchvision's DenseNet.forward; the classifier is the head
        m.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(m.classifier.in_features, n_out))
        m.head_name = "classifier"
    elif name == "mcf_densenet121":
        m = MCFDenseNet121(n_out, pretrained, dropout, memory_efficient)
    elif name == "swin_t":
        m = SwinT(n_out, pretrained, dropout, memory_efficient)
    else:
        raise ValueError(f"unknown model {name!r}")
    if label_mean is not None:
        # Start each output at its average training label: sigmoid(bias) = mean, so the scaled MSE
        # starts near 1 per value instead of (0.5 - 0)^2 / var (about 800 for noise).
        bias = torch.tensor([_logit(p) for p in label_mean])
        with torch.no_grad():
            for lin in output_layers(m):
                lin.bias.copy_(bias)
    return m


def output_layers(m):
    """The Linear layers that produce the 9 scores (one for DenseNet / Swin-T, five for MCF)."""
    if isinstance(m, MCFDenseNet121):
        return [m.heads[k][-1] for k in ("rgb", "hsv", "lab", "feature")] + [m.heads["final"]]
    return [head_of(m)[-1]]


def head_of(m):
    return getattr(m, m.head_name)


def parameter_counts(m):
    total = sum(p.numel() for p in m.parameters())
    head = sum(p.numel() for p in head_of(m).parameters())
    return {"params_total": total, "params_backbone": total - head, "params_head": head}


def param_groups(m, lr_backbone, lr_head, weight_decay):
    """AdamW groups: backbone / head learning rates; no weight decay on norm weights, biases and Swin's
    relative-position-bias tables (C2)."""
    head_ids = {id(p) for p in head_of(m).parameters()}
    groups = {}
    for name, p in m.named_parameters():
        if not p.requires_grad:
            continue
        part = "head" if id(p) in head_ids else "backbone"
        decay = p.ndim > 1 and "relative_position_bias_table" not in name
        key = (part, decay)
        if key not in groups:
            groups[key] = {"params": [], "name": f"{part}{'' if decay else '_no_decay'}",
                           "lr": lr_head if part == "head" else lr_backbone,
                           "weight_decay": weight_decay if decay else 0.0}
        groups[key]["params"].append(p)
    out = [groups[k] for k in sorted(groups)]
    for g in out:
        g["base_lr"] = g["lr"]
    return out


# ---------------------------------------------------------------- colour spaces (MCF-style, C1)

def rgb_to_hsv(rgb):
    """[B, 3, H, W] RGB in [0, 1] -> HSV in [0, 1], as Pillow's convert("HSV") / colorsys (ties: R, then G, then B)."""
    r, g, b = rgb.unbind(1)
    maxc, _ = rgb.max(1)
    minc, _ = rgb.min(1)
    d = maxc - minc
    safe = torch.where(d > 0, d, torch.ones_like(d))
    rc, gc, bc = (maxc - r) / safe, (maxc - g) / safe, (maxc - b) / safe
    h = torch.where(r == maxc, bc - gc, torch.where(g == maxc, 2.0 + rc - bc, 4.0 + gc - rc))
    h = torch.where(d > 0, torch.remainder(h / 6.0, 1.0), torch.zeros_like(h))
    s = torch.where(maxc > 0, d / torch.where(maxc > 0, maxc, torch.ones_like(maxc)), torch.zeros_like(maxc))
    return torch.stack([h, s, maxc], 1)


# sRGB (linear) -> XYZ relative to D50, the ICC matrix (Pillow ImageCms' LAB profile is D50)
_RGB2XYZ_D50 = ((0.4360747, 0.3850649, 0.1430804),
                (0.2225045, 0.7168786, 0.0606169),
                (0.0139322, 0.0971045, 0.7141733))
_WHITE_D50 = (0.9642, 1.0, 0.8249)


def rgb_to_lab(rgb):
    """[B, 3, H, W] sRGB in [0, 1] -> CIELAB (D50) scaled to [0, 1]: L / 100, (a + 128) / 255, (b + 128) / 255.
    Continuous on purpose: MCF-Net's Pillow LAB stores negative a / b as wrapped bytes ."""
    lin = torch.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    r, g, b = lin.unbind(1)
    xyz = [(m[0] * r + m[1] * g + m[2] * b) / w for m, w in zip(_RGB2XYZ_D50, _WHITE_D50)]
    eps, kappa = 216 / 24389, 24389 / 27
    fx, fy, fz = [torch.where(t > eps, t.clamp(min=eps) ** (1 / 3), (kappa * t + 16) / 116) for t in xyz]
    L, a, bb = 116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)
    return torch.stack([L / 100, (a + 128) / 255, (bb + 128) / 255], 1)


class MCFDenseNet121(nn.Module):
    """3 DenseNet121 branches (RGB / HSV / LAB) as in the official MCF-Net code (networks/densenet_mcf.py):
    branch heads on pooled features, feature fusion Linear(3 x 1024 -> 9) on the concatenated pooled features,
    prediction fusion Linear(4 x 9 -> 9) on the 4 sigmoid predictions. The input is the ImageNet-normalised RGB batch
    (engine.Batch); HSV / LAB are computed here in float32 and normalised with the same ImageNet mean / std."""

    SPACES = ("rgb", "hsv", "lab")

    def __init__(self, n_out, pretrained=True, dropout=0.2, memory_efficient=False):
        super().__init__()
        weights = torchvision.models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        self.branches = nn.ModuleDict({s: torchvision.models.densenet121(
            weights=weights, memory_efficient=memory_efficient).features for s in self.SPACES})
        nf = 1024
        self.heads = nn.ModuleDict({s: nn.Sequential(nn.Dropout(dropout), nn.Linear(nf, n_out)) for s in self.SPACES})
        self.heads["feature"] = nn.Sequential(nn.Dropout(dropout), nn.Linear(3 * nf, n_out))
        self.heads["final"] = nn.Linear(4 * n_out, n_out)
        nn.init.zeros_(self.heads["final"].weight)  # starts at the label mean (build_model sets the bias)
        self.head_name = "heads"
        self.checkpoint_branches = memory_efficient  # C1: peak memory of about one branch
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

    def _pooled(self, space, x):
        f = torch.relu(self.branches[space](x))
        return torch.flatten(nn.functional.adaptive_avg_pool2d(f, 1), 1)

    def forward(self, x):
        with torch.autocast(device_type=x.device.type, enabled=False):
            rgb = (x.float() * self.std + self.mean).clamp(0, 1)
            inputs = {"rgb": x,
                      "hsv": (rgb_to_hsv(rgb) - self.mean) / self.std,
                      "lab": (rgb_to_lab(rgb) - self.mean) / self.std}
        if x.is_contiguous(memory_format=torch.channels_last):
            inputs = {s: v.contiguous(memory_format=torch.channels_last) for s, v in inputs.items()}
        feats = {}
        for s in self.SPACES:
            if self.checkpoint_branches and self.training:
                feats[s] = checkpoint(self._pooled, s, inputs[s], use_reentrant=False)
            else:
                feats[s] = self._pooled(s, inputs[s])
        scores = [self.heads[s](feats[s]) for s in self.SPACES]
        scores.append(self.heads["feature"](torch.cat([feats[s] for s in self.SPACES], 1)))
        final = self.heads["final"](torch.sigmoid(torch.cat(scores, 1)))
        if not self.training:
            return final
        return torch.stack(scores + [final], 1)


class SwinT(nn.Module):
    """torchvision swin_t with a 9-value head. Same submodule names as torchvision (features, norm, ..., head),
    so its ImageNet weights and our checkpoints load directly. Window attention pads internally, so 512 x 672 works
    without interpolating weights (C2). memory_efficient: recompute each block in the backward pass."""

    def __init__(self, n_out, pretrained=True, dropout=0.2, memory_efficient=False):
        super().__init__()
        weights = torchvision.models.Swin_T_Weights.IMAGENET1K_V1 if pretrained else None
        base = torchvision.models.swin_t(weights=weights)
        self.features, self.norm, self.permute = base.features, base.norm, base.permute
        self.avgpool, self.flatten = base.avgpool, base.flatten
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(base.head.in_features, n_out))
        self.head_name = "head"
        self.checkpoint_blocks = memory_efficient

    def forward(self, x):
        for stage in self.features:
            if self.checkpoint_blocks and self.training and isinstance(stage, nn.Sequential):
                for blk in stage:
                    x = checkpoint(blk, x, use_reentrant=False)
            else:
                x = stage(x)
        return self.head(self.flatten(self.avgpool(self.permute(self.norm(x)))))
