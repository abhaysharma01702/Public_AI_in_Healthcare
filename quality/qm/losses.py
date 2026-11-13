"""Loss of the quality module (section 2 and A).

total = (1/9) sum_k MSE_k / var_k  +  rank_weight * RankNet

- MSE on sigmoid(score); var_k = variance of value k over all training labels (clean included).
- RankNet per defect k over all ordered pairs (i, j) in the batch where image i should rank above j:
    (a) defect present in i, absent in j (label_i > 0 = label_j), whatever the size (point 7b), or
    (b) present in both and label_i - label_j >= min_diff_k (= 5% of the defect's 95th-percentile
        training label, A).
  pair loss = -log sigmoid(s_i - s_j) on the raw scores; averaged per defect, then over the defects
  that have at least one pair in the batch (each defect counts equally, like the variance scaling).
"""
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def label_stats(train_df, values, rank_min_frac):
    """Per-value statistics from the training labels (saved with the run)."""
    y = train_df[values].to_numpy(np.float64)
    present = [y[:, k][y[:, k] > 0] for k in range(len(values))]
    return {
        "values": list(values),
        "n_train": int(len(y)),
        "mean": y.mean(0).tolist(),
        "var": y.var(0, ddof=1).tolist(),
        "p95_present": [float(np.quantile(p, 0.95)) for p in present],
        "min_diff": [float(rank_min_frac * np.quantile(p, 0.95)) for p in present],
        "rank_min_frac": rank_min_frac,
    }


def rank_pairs(y, min_diff):
    """Boolean [B, B, K]: True where image i must score above image j for defect k."""
    yi, yj = y[:, None, :], y[None, :, :]
    present_vs_absent = (yi > 0) & (yj == 0)
    both_present = (yi > 0) & (yj > 0) & (yi - yj >= min_diff)
    return present_vs_absent | both_present


def ranknet(scores, y, min_diff):
    pairs = rank_pairs(y, min_diff)
    pair_loss = F.softplus(-(scores[:, None, :] - scores[None, :, :]))  # -log sigmoid(s_i - s_j)
    n = pairs.sum((0, 1))  # pairs per defect
    per_defect = (pair_loss * pairs).sum((0, 1)) / n.clamp(min=1)
    has = n > 0
    if not has.any():
        return scores.sum() * 0.0
    return per_defect[has].mean()


class QualityLoss(nn.Module):
    def __init__(self, var, min_diff, rank_weight):
        super().__init__()
        self.register_buffer("inv_var", 1.0 / torch.tensor(var, dtype=torch.float32))
        self.register_buffer("min_diff", torch.tensor(min_diff, dtype=torch.float32))
        self.rank_weight = rank_weight

    def forward(self, scores, y):
        scores = scores.float()
        mse = ((torch.sigmoid(scores) - y) ** 2 * self.inv_var).mean()
        rank = ranknet(scores, y, self.min_diff)
        return mse + self.rank_weight * rank, {"mse": mse.detach(), "rank": rank.detach()}


class MultiOutputLoss(nn.Module):
    """MCF-style training loss (C1): sum_i w_i * QualityLoss(output_i) over scores [B, n_outputs, 9].
    Scores [B, 9] (eval mode: the final output only) get the plain loss, so validation loss and model selection
    use the final prediction, as for the other models."""

    def __init__(self, base, weights):
        super().__init__()
        self.base = base
        self.weights = tuple(weights)

    def forward(self, scores, y):
        if scores.ndim == 2:
            return self.base(scores, y)
        assert scores.shape[1] == len(self.weights), (scores.shape, self.weights)
        total, mse, rank = 0.0, 0.0, 0.0
        for i, w in enumerate(self.weights):
            loss, parts = self.base(scores[:, i], y)
            total, mse, rank = total + w * loss, mse + w * parts["mse"], rank + w * parts["rank"]
        return total, {"mse": mse, "rank": rank}
