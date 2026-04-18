# III. Methodology

## A. Dataset

The SIIM-ISIC 2020 dataset (`nroman/melanoma-external-malignant-256`) contains
~37k dermoscopic images (256×256, **13.6% malignant**) with metadata: `age_approx`
(continuous), `sex` (binary), `anatom_site_general_challenge` (6 categories).
An 80/20 stratified split (seed 42) preserves class ratio.

## B. Preprocessing and Augmentation

**Images (train):** Albumentations (CLAHE, blur, 90° snaps) then torchvision v2
(flip, affine ±15°, colour jitter, random erasing, ImageNet normalisation).
**Images (val):** resize + normalise only. **Metadata:** median imputation +
z-score for age; mode imputation + one-hot for categoricals → **14-d** vector.
SHAP analysis is reported in Section IV-F.

## C. Architecture

A dual-branch late-fusion model: (1) a pretrained CNN backbone (EfficientNet-B0
→ 1280-d, DenseNet-121 → 1024-d, ResNet-50 → 2048-d) with head removed;
(2) a two-layer MLP (14→128→64, BN+ReLU+Dropout 0.3) for metadata. Embeddings
are concatenated and projected through a linear head to one logit.

## D. Loss and Training

Sigmoid focal loss: $FL(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$ with
α=0.864, γ=2.0.

| Hyperparameter    | Baseline (A)       | Configs B–D (IV-C)     |
|-------------------|--------------------|------------------------|
| Optimiser         | Adam               | Adam / AdamW           |
| Learning rate     | 1×10⁻⁴ (fixed)    | 1×10⁻⁴ + cosine       |
| Weight decay      | 0                  | 0 / 1×10⁻⁵            |
| Focal γ           | 2.0                | 2.0 / 1.5             |
| Batch / Epochs    | 32 / 20            | 32 / 20               |

Best validation F1 checkpoint is retained per configuration.

## E. TTA

Six geometric transforms (original, h-flip, v-flip, 90°/180°/270° rotation)
averaged at inference. Impact evaluated in Section IV-B.

## F. Evaluation

Accuracy, recall, specificity, PPV, F1 on the validation split. Recall is the
primary metric: missed melanomas carry higher clinical cost than false positives.

---

[← II. Literature Review](03_literature_review.md) | [Next → IV. Experiments](05_experiments.md)
