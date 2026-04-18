# Abstract

This work presents a dual-branch deep learning system fusing dermoscopic images
with patient metadata (age, sex, anatomical site) for melanoma classification on
the SIIM-ISIC 2020 dataset (~13.6% malignant). Three CNN backbones are compared
under sigmoid focal loss (α=0.864). A four-configuration hyperparameter search
identifies cosine annealing as the most impactful change (F1: 0.8932→0.8984,
FN reduced 12.5%). An image-only ablation confirms metadata fusion adds +1.9%
recall; SHAP analysis identifies anatomical site and age as dominant predictors.
With six-transform TTA the system achieves F1=0.9134, recall=91.48%,
specificity=98.62%. EigenCAM validates ABCDE-aligned attention and a
Mahalanobis OOD detector flags non-dermoscopic inputs.

*Index Terms*—melanoma, focal loss, metadata fusion, SHAP, TTA.

---

[Next → II. Introduction](02_introduction.md)
