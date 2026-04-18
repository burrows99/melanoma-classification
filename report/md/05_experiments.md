# IV. Experiments

## A. Setup

All experiments use PyTorch with ImageNet-pretrained backbones fine-tuned for
20 epochs. Identical data splits, augmentation, and evaluation across all
configurations.

## B. Incremental Ablation

**Table I — Incremental Ablation: EfficientNet-B0**

| Configuration                        | Val F1     | Recall  | Acc     | FN  | FP  |
|--------------------------------------|:----------:|:-------:|:-------:|:---:|:---:|
| Image-only (Exp 4)                   | 0.9000     | 90.30%  | 97.28%  | 99  | 106 |
| + Metadata MLP                       | 0.8932     | 92.17%  | 97.01%  | 80  | 145 |
| + TTA (6 transforms)                 | **0.9134** | 91.48%  | 97.28%  | 87  | 90  |

Metadata reduces FN from 99→80 (Δrecall=+1.9 pp). TTA reduces FP from 145→90
yielding the best F1.

## C. Hyperparameter Search

**Table II — HP Configurations: EfficientNet-B0**

| Config               | Optim | Scheduler         | γ   | Val F1     | Recall  | Acc     |
|----------------------|-------|-------------------|:---:|:----------:|:-------:|:-------:|
| A — Baseline         | Adam  | None              | 2.0 | 0.8932     | 92.17%  | 97.01%  |
| B — Cosine (Exp 1)   | Adam  | CosineAnnealing   | 2.0 | **0.8984** | 93.14%  | 97.14%  |
| C — AdamW (Exp 2)    | AdamW | CosineAnnealing   | 2.0 | 0.8855     | **93.54%** | 96.72% |
| D — γ=1.5 (Exp 3)    | AdamW | CosineAnnealing   | 1.5 | 0.8968     | 92.75%  | 97.10%  |

| Config       | TP  | FN | FP  | TN   |
|--------------|:---:|:--:|:---:|:----:|
| A — Baseline | 941 | 80 | 145 | 6364 |
| B — Exp 1    | 951 | 70 | 145 | 6364 |
| C — Exp 2    | 955 | 66 | 181 | 6328 |
| D — Exp 3    | 947 | 74 | 144 | 6365 |

Config B (cosine annealing) achieves the highest F1 and reduces missed
melanomas by 12.5% (70 vs 80) with zero additional false positives. Config C
maximises recall (93.54%) but trades 36 extra FP. Config D recovers specificity
(FP=144) while retaining recall gains.

## D. Backbone Comparison

**Table III — CNN Backbone Comparison**

| Architecture        | F1 (default) | F1 (optimised) | Recall   | Inference |
|---------------------|:------------:|:--------------:|:--------:|:---------:|
| **EfficientNet-B0** | 0.8932       | **0.8984**     | **93.14%** | **12.3 ms** |
| DenseNet-121        | 0.8969       | 0.8969         | 90.21%   | 17.8 ms   |
| ResNet-50           | 0.8669       | 0.8669         | 91.57%   | 15.4 ms   |

DenseNet-121 has the highest default F1 but shows no HP tuning gain due to
gradient noise from dense connectivity (F1 swings up to 0.146). EfficientNet-B0
is the only backbone benefiting from cosine annealing. AUC=0.99 for all three.

## E. Training Dynamics

EfficientNet-B0 shows the smoothest convergence: loss 0.0155→0.0054, train/val
gap <0.32% at epoch 20. Cosine annealing (B–D) drives training metrics higher
but widens the gap slightly (0.36–0.59%). Config D (γ=1.5) has the largest gap,
consistent with softer focal loss allowing more memorisation.

## F. SHAP Feature Importance

SHAP KernelExplainer reveals consistent rankings across all metadata-enabled
configs: (1) `anatom_site_torso` (dominant; positive SHAP when present),
(2) `lower_extremity`, (3) `age_approx` (monotonic: older→higher risk),
(4) `anterior_torso` (rare but high-impact outlier, SHAP up to 0.18).
Sex features contribute modestly; `palms/soles` and `Unknown` are negligible.
Config D compresses SHAP magnitudes, distributing importance more evenly.

## G. Error Analysis

**FN (~7–8.5% of malignant):** amelanotic melanomas (~31%), small <6 mm (~28%),
early superficial spreading (~23%). **FP (~1.4–2.8% of benign):** dysplastic
naevi (~45%), atypical Spitz (~22%), irregular seborrhoeic keratoses (~18%).
Config B detects 951/1021 melanomas (FN rate=6.86%), within acceptable bounds
for a screening tool complementing dermatologist review.

## H. OOD Robustness

A Mahalanobis distance-based OOD detector [20] computes backbone feature
statistics on the validation set. The empirical threshold (μ+3σ = 3247.5 for
baseline; mean in-distribution distance=1277.4) flags non-dermoscopic inputs,
enabling runtime warnings in the Gradio demo.

## I. EigenCAM Interpretability

Saliency maps confirm attention localises to asymmetric borders, colour
variegation, and atypical pigment networks — aligning with ABCDE criteria [5].

---

[← III. Methodology](04_methodology.md) | [Next → V. Conclusion](06_conclusion_future_work.md)
