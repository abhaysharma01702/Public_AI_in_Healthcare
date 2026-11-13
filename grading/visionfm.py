"""VisionFM slit-lamp encoder, frozen, for the Reti-Pioneer-style grader.
VisionFM (ABILab-CUHK, github.com/ABILab-CUHK/VisionFM; research / educational use only) pre-trained an iBOT ViT-Base/16
at 224 x 224 per modality. The slit-lamp checkpoint ("Silt Lamp" row of the README, Google Drive) is 1.5 GB (student,
teacher, optimiser); weights/visionfm/VFM_SlitLamp_teacher_backbone.pth keeps only the teacher backbone (as the
VisionFM evaluation code: checkpoint key "teacher", prefixes "module." / "backbone." removed).
The ViT is written out here (the DINO / iBOT vision_transformer layout: pre-norm blocks, LayerNorm eps 1e-6,
qkv with bias, GELU), so the state dict loads with strict=True and no extra package is needed.
Preprocessing as VisionFM: resize to 224 x 224 (no crop), ToTensor, Normalize with the SlitLamp statistics of
VisionFM utils.get_stats.
Features (DINO linear-evaluation layout for ViT-B, n_last_blocks 1 + avgpool): the final-norm CLS token
joined with the mean of the final-norm patch tokens -> 1,536 values.
Fine-tuning (train_grading.py --arch visionfm): any input size divisible by 16 (e.g. 384 x 512); the
14 x 14 position embedding of the checkpoint is resized bicubically to the new patch grid (as DINO / iBOT do);
grad_ckpt recomputes each block in the backward pass to save GPU memory.
"""
import os

import torch
import torch.nn as nn
import torch.utils.checkpoint

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS = os.path.join(ROOT, "weights", "visionfm", "VFM_SlitLamp_teacher_backbone.pth")
SIZE = 224
MEAN = torch.tensor([0.5556667, 0.40288574, 0.38857886]).view(1, 3, 1, 1)
STD = torch.tensor([0.26516676, 0.23311588, 0.23903583]).view(1, 3, 1, 1)
FEAT_DIM = 1536


class Attention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        b, n, c = x.shape
        q, k, v = self.qkv(x).reshape(b, n, 3, self.heads, c // self.heads).permute(2, 0, 3, 1, 4)
        x = nn.functional.scaled_dot_product_attention(q, k, v)
        return self.proj(x.transpose(1, 2).reshape(b, n, c))


class Mlp(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.fc1, self.act, self.fc2 = nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.norm1, self.attn = nn.LayerNorm(dim, eps=1e-6), Attention(dim, heads)
        self.norm2, self.mlp = nn.LayerNorm(dim, eps=1e-6), Mlp(dim, 4 * dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class PatchEmbed(nn.Module):
    def __init__(self, dim, patch):
        super().__init__()
        self.proj = nn.Conv2d(3, dim, patch, patch)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


class ViTBase(nn.Module):
    def __init__(self, dim=768, depth=12, heads=12, patch=16, size=(SIZE, SIZE), grad_ckpt=False):
        super().__init__()
        self.grid = (size[0] // patch, size[1] // patch)
        self.grad_ckpt = grad_ckpt
        self.patch_embed = PatchEmbed(dim, patch)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.grid[0] * self.grid[1] + 1, dim))
        self.blocks = nn.ModuleList([Block(dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, x):
        """x: normalised images [B, 3, 224, 224] -> features [B, 1536] (CLS + mean patch token)."""
        x = self.patch_embed(x)
        x = torch.cat([self.cls_token.expand(len(x), -1, -1), x], 1) + self.pos_embed
        for blk in self.blocks:
            if self.grad_ckpt and self.training:
                x = torch.utils.checkpoint.checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)
        x = self.norm(x)
        return torch.cat([x[:, 0], x[:, 1:].mean(1)], 1)


def resize_pos_embed(pos, grid):
    """[1, 1 + 14*14, D] -> [1, 1 + gh*gw, D]: CLS position kept, patch positions resized bicubically."""
    cls, pat = pos[:, :1], pos[:, 1:]
    n = int(round(pat.shape[1] ** 0.5))
    if (n, n) == tuple(grid):
        return pos
    pat = pat.reshape(1, n, n, -1).permute(0, 3, 1, 2)
    pat = nn.functional.interpolate(pat, size=grid, mode="bicubic", align_corners=False)
    return torch.cat([cls, pat.permute(0, 2, 3, 1).reshape(1, grid[0] * grid[1], -1)], 1)


def load_encoder(path=WEIGHTS, size=(SIZE, SIZE), grad_ckpt=False):
    model = ViTBase(size=size, grad_ckpt=grad_ckpt)
    sd = torch.load(path, map_location="cpu", weights_only=True)
    sd["pos_embed"] = resize_pos_embed(sd["pos_embed"], model.grid)
    model.load_state_dict(sd, strict=True)
    return model.eval()
