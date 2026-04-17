# Abstract

Melanoma is the most lethal form of skin cancer, yet early detection offers
near-complete survivability. This work presents a dual-branch deep learning
system fusing dermoscopic images with patient metadata (age, sex, anatomical
site) on the SIIM-ISIC 2020 dataset. Three backbones—EfficientNet-B0,
DenseNet-121, and ResNet-50—are compared under identical conditions with severe
class imbalance (~13.6% malignant) handled by sigmoid focal loss (α=0.864,
γ=2.0). Key contributions: (1) principled derivation of focal loss α from class
proportions; (2) metadata fusion improving recall by ~2.8% over an image-only
baseline; (3) test-time augmentation (TTA) across six geometric transforms for
conservative clinical inference. The optimal configuration (EfficientNet-B0 +
metadata + TTA) achieves F1=0.9134, recall=91.48%, specificity=98.62%. EigenCAM
maps confirm model attention localises to ABCDE-relevant lesion features.

*Index Terms*—melanoma classification, focal loss, metadata fusion, TTA.

---

[Next → II. Introduction](02_introduction.md)
