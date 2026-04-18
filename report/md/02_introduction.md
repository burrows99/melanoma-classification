# I. Introduction

Melanoma causes over 75% of skin cancer deaths despite representing fewer than
5% of cases [1]. Five-year survival exceeds 98% at stage I but falls below 25%
for metastatic disease [2]. Dermoscopy improves sensitivity by 10–27 pp [3],
yet inter-observer variability persists. Esteva et al. [5] showed a fine-tuned
CNN matched 21 dermatologists, and the ISIC 2020 challenge [7] released
structured metadata alongside dermoscopic images, motivating late-fusion
approaches that incorporate patient context.

A further challenge is severe class imbalance: at ~13.6% positive rate, a
trivial all-benign predictor achieves 86.4% accuracy [13].

**Contributions:**
1. Dual-branch fusion (CNN + metadata MLP) with focal loss α=0.864 derived
   from class proportions.
2. Controlled three-backbone comparison (EfficientNet-B0, DenseNet-121,
   ResNet-50) with inference time and convergence analysis.
3. Four-configuration HP search isolating scheduler, optimiser, and focal γ.
4. +1.9% recall from metadata fusion (measured ablation) with per-feature SHAP.
5. Error analysis, EigenCAM interpretability, and Mahalanobis OOD detection.

**Out of scope:** ViTs [21, 22] (require more data to compensate for lack of
CNN inductive biases [21]), GAN augmentation, ensembles, multi-class diagnosis,
clinical validation, and the ISIC private test set.

---

[← Abstract](01_abstract.md) | [Next → II. Literature Review](03_literature_review.md)
