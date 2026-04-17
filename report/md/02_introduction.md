# I. Introduction

Melanoma accounts for fewer than 5% of skin cancer cases yet more than 75% of
skin cancer deaths [1]. Five-year survival exceeds 98% at stage I but falls
below 25% for metastatic disease, making early and accurate diagnosis one of
the most clinically impactful interventions available [2]. Dermoscopy improves
diagnostic sensitivity by 10–27 percentage points over naked-eye examination [3],
yet accurate interpretation requires years of specialist training and
inter-observer variability remains a well-documented clinical challenge.

The landmark result of Esteva et al. [5], who showed that a fine-tuned
Inception-v3 matched the accuracy of 21 board-certified dermatologists,
established deep learning as a clinically credible approach. The ISIC 2020
challenge [7] advanced this by releasing per-image patient identifiers and
structured metadata (age, sex, anatomical site), acknowledging that
dermatologists integrate patient history when forming diagnoses. A persistent
limitation of image-only classifiers is that they discard this routinely
available context; late-fusion of CNN embeddings with tabular patient data
provides a principled approach to incorporating these priors.

A further structural challenge is severe class imbalance. In the ISIC 2020
dataset, malignant samples represent only ~13.6% of images, creating a 6.4:1
imbalance that standard cross-entropy handles poorly, yielding high accuracy
but low sensitivity classifiers [13].

This work makes the following contributions:
1. A dual-branch fusion model combining a pretrained CNN backbone with a
   metadata MLP, trained end-to-end using sigmoid focal loss with
   mathematically derived α=0.864.
2. A controlled comparison of EfficientNet-B0, DenseNet-121, and ResNet-50
   under identical conditions, including inference time and convergence analysis.
3. Demonstration that metadata fusion improves recall by ~2.8% over an
   image-only baseline, with a per-feature clinical interpretation (age, sex,
   anatomical site).
4. Structured error analysis of FN/FP patterns, EigenCAM visualisations
   validating ABCDE alignment, and discussion of OOD robustness limitations.

**Out of scope:**
- Multi-class or multi-label skin lesion diagnosis beyond binary melanoma
  detection.
- Transformer-based vision backbones (e.g., Vision Transformer, Swin) or
  GAN-based synthetic data augmentation.
- Ensemble methods or model distillation; each backbone is evaluated
  independently as a single model.
- Clinical validation, regulatory approval, or deployment to production
  healthcare systems.
- Evaluation on the official ISIC private test set or any external clinical
  cohort beyond the described validation split.

The remainder of this paper is organised as follows. Section II surveys related
work. Section III details the methodology. Section IV presents experimental
results and observations. Section V concludes with directions for future work.

---

[← Abstract](01_abstract.md) | [Next → II. Literature Review](03_literature_review.md)
