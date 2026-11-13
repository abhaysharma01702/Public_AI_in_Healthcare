# Quality-Aware Grading of Slit-Lamp Images

Course project, AI for Healthcare (IIIT Hyderabad), by Abhay Sharma and Shaik Areesh.

Slit-lamp eye photos taken in clinics are often blurred, badly exposed or partly blocked. We asked whether a disease
grader becomes more reliable on such photos when it is trained on degraded photos, and when it is also told how bad
the photo is. The project has three parts:

1. **Quality module:** a network that scores a photo on 9 defects: defocus blur, motion blur, overexposure,
   underexposure, uneven illumination, haze, glare, occlusion and noise. Each output is a severity between 0 and 1.
2. **Disease graders** for two tasks: the Nuclear Cataract grade (6 grades) and the SUSTech-SYSU corneal-ulcer
   category (3 classes), trained under the conditions below.
3. **Explainability:** probes, head-weight analysis and saliency maps that explain the graders' behaviour.

| Condition | Training photos | Extra input |
|---|---|---|
| A | clean only | none |
| B | clean + degraded | none |
| C | clean + degraded | the 9 quality scores |
| C_shuffled (test-time check) | as C | quality scores shuffled between photos |

All defects are synthetic. They are added by our own code with a known strength, so every label is exact. The test
defects come from a separate implementation that was never used during development.

## Results

### Quality module

Fresh test set: 3,324 photos, with defects made by the unseen implementation.

| Model | AUROC (is the defect present?) | Spearman (how severe?) |
|---|---|---|
| DenseNet121 | 0.883 | 0.699 |
| DenseNet121, extended training data | 0.939 | 0.722 |
| MCF-style DenseNet (RGB / HSV / LAB) | 0.941 | 0.738 |
| **Swin-T, 512 × 672 (final)** | **0.953** | **0.767** |

### Graders

QWK, mean of 3 seeds, fine-tuned VisionFM ViT-B/16 at 384 × 512.

| Task | Test | A | B | C | C_shuffled |
|---|---|---|---|---|---|
| Nuclear Cataract | clean | 0.604 | 0.622 | 0.632 | 0.633 |
| Nuclear Cataract | degraded | 0.542 | 0.582 | 0.572 | 0.570 |
| SUSTech-SYSU | clean | 0.685 | 0.570 | 0.587 | 0.574 |
| SUSTech-SYSU | degraded | 0.482 | 0.504 | 0.500 | 0.503 |

- **Training on degraded photos (B) helps on degraded photos.** Giving the quality scores (C) adds nothing:
  C ≈ C_shuffled. We saw the same across six grader designs.
- **Final Nuclear Cataract grader:** a regression on the raw NC value, a 5-seed ensemble, and cut points tuned on
  validation. On degraded photos it reaches QWK 0.663, against 0.573 for the earlier classifier (+0.090, 95% CI
  [+0.013, +0.167]).
- **Final SUSTech-SYSU grader:** condition A, a 3-seed ensemble. Clean QWK 0.719, degraded 0.524.

### Why: explainability (`results/explainability/`)

- **The grader ignores the quality input.** The head is linear, so its output splits exactly into an image part and
  a quality part. The quality part accounts for under 0.1% of the output variance, and removing it changes 1–2% of
  the predicted grades.
- **The image features already encode the defect.** A linear probe on the grader's features identifies the defect
  type 88–90% of the time (chance 11%). This holds even for a grader that never saw a degraded photo, so an explicit
  quality vector has nothing new to add.
- **Accuracy loss follows feature drift.** How far a defect moves a photo's features from its clean version
  predicts how far the prediction moves (Spearman 0.69 for Nuclear Cataract, 0.48 for SUSTech-SYSU).
- **Occlusion is the most harmful defect.** Nuclear Cataract QWK is 0.28 under occlusion, against 0.68 on clean
  photos. The Nuclear Cataract grader is otherwise robust (0.67–0.78 for every other defect). The SUSTech-SYSU grader
  degrades steadily with strength (0.66 at strength 0.2, 0.40 at 1.0).
- **Where the grader looks.** For Nuclear Cataract, the saliency maps fall on the lens nucleus, and every defect
  except occlusion leaves them in place. The correlation with the clean map is 0.76–0.93 for those defects and
  0.22 under occlusion. For SUSTech-SYSU, the maps spread over the corneal surface, the limbus and the specular
  reflection. Blur moves them more (0.66–0.69), and occlusion most (0.18).
- **Rare grades are the hardest.** NC1 has 44 training photos and is never predicted. SUSTech-SYSU's smallest class
  (70 training photos) is the first lost on degraded photos.

## Repository layout

| Folder | Contents |
|---|---|
| `data_prep/` | per-dataset master tables, eye- and patient-level splits, duplicate checks, 512 px resizing, quality screening |
| `quality/` | defect generators (training, held-out and fresh test implementations), quality module (`qm/`), training and evaluation, Slurm job scripts |
| `grading/` | grading data, grader training (DenseNet121 / VisionFM), prediction, evaluation, post-hoc analyses, explainability |
| `figures/` | figure scripts and figures |
| `results/` | result tables and plots |
| `Docs/Poster/` | showcase poster |

## Running

```bash
pip install -r requirements.txt

# quality module
python quality/make_quality_data.py
python quality/train.py --config quality/configs/swin_t.yaml --data_root <quality data> --run_dir runs/quality/swin_t
python quality/evaluate.py --run_dir runs/quality/swin_t --data_root <quality data>

# graders
python grading/make_grading_data.py --task nuclear_cataract
python grading/train_grading.py --help
python grading/evaluate_grading.py --runs <run folder>

# explainability
python grading/explain_features.py --runs <run folders> --data_root "Data/Grading Data/nuclear_cataract"
python grading/explain_analysis.py
python grading/explain_saliency.py --task nuclear_cataract --run <run folder>
```

Each script's docstring describes its inputs, outputs and options. The `*.sbatch` files are the Slurm jobs we ran;
replace `<your_account>` and `<your_qos>` with your cluster's values.

## Data

The datasets are not included; please obtain them from their original sources under their own licences. Expected
location: `Data/`. We used SLID, SLID-E, SLIT-Net, SLIT-Net v2, SDCTrans, the Infectious Keratitis dataset, the
Nuclear Cataract dataset and SUSTech-SYSU. Trained model weights are not included because of their size.
