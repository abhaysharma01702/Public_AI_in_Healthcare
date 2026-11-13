"""Quality-module training code.

data.py     QualityDataset, SourceGroupedBatchSampler, set loaders (val / test / stress / real check)
model.py    build_model (DenseNet121 baseline)
losses.py   variance-scaled MSE + per-defect RankNet (A)
metrics.py  AUROC / MAE / Spearman / cross-talk, per dataset and lighting, bootstrap CIs (B)
engine.py   training epoch, evaluation pass, EMA, checkpoints
"""
VALUES = ["defocus_blur", "motion_blur", "overexposure", "underexposure", "uneven_illumination",
          "haze", "glare", "occlusion", "noise"]  # same order as quality/degradations.py VALUES

# Lighting per dataset; Infectious Keratitis mixes all three without per-image labels.
LIGHTING = {
    "SLID": "diffuse_white",
    "SLID-E": "cobalt_blue",
    "SUSTech-SYSU": "cobalt_blue",
    "Nuclear Cataract": "slit_beam",
    "Infectious Keratitis": "mixed",
}
