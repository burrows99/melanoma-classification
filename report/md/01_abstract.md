# Abstract

Melanoma is the most lethal form of skin cancer, responsible for the majority
of skin cancer deaths despite constituting fewer than 5% of cases. Early
automated detection from dermoscopic images offers a scalable path to improved
patient outcomes, yet image-only classifiers discard routinely available clinical
context. This paper presents a multi-modal deep learning system that fuses
dermoscopic image features from a pretrained convolutional neural network (CNN)
with structured patient metadata—age, biological sex, and anatomical site—via a
dual-branch architecture trained end-to-end on the SIIM-ISIC 2020 dataset. Three
backbone architectures, EfficientNet-B0, DenseNet-121, and ResNet-50, are
systematically compared under identical training conditions. Severe class
imbalance (~13.6% malignant) is addressed using sigmoid focal loss with a
mathematically derived class-proportional weighting (α=0.864). Key contributions
include: (1) a principled derivation of the focal loss α parameter directly from
class distribution statistics, which optimises recall for the minority malignant
class; (2) a dual-branch architecture combining EfficientNet-B0 with a metadata
MLP that improves recall by approximately 2.8% compared with image-only
approaches; and (3) an implemented test-time augmentation (TTA) pipeline
applying six geometric transforms for conservative probability estimation suited
to clinical screening. EfficientNet-B0 achieves the best single-model
performance: validation F1 of 0.9087, accuracy of 97.48%, and recall of 92.65%,
compared with a winning ISIC 2020 ensemble AUC of 0.9490. EigenCAM saliency
maps confirm that the model attends to clinically relevant lesion features
(asymmetric borders, colour variegation) aligned with the ABCDE diagnostic
criteria.

---

| | |
|---|---|
| | [Next → II. Introduction](02_introduction.md) |
