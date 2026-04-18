# II. Literature Review

## A. Backbones

The ISIC 2020 challenge [7] released 33,126 dermoscopic images with patient
metadata. The winning ensemble of EfficientNets [8] achieved AUC=0.9490.
**EfficientNet-B0** [9] (5.3M params) applies compound scaling for strong
transfer learning. **DenseNet-121** [10] uses dense connectivity validated for
medical imaging by Rajpurkar et al. [11]. **ResNet-50** [12] remains the
canonical baseline. ViTs [21, 22] show promise on large datasets but require
substantially more data due to the absence of translation equivariance [21];
CNNs with ImageNet pretraining remain pragmatic at this scale.

## B. Class Imbalance

Focal loss [13] modulates cross-entropy by $(1-p_t)^\gamma$, concentrating
gradient on hard cases. Setting α = 1 − 0.136 = 0.864 is derived directly
from class statistics.

## C. Metadata, TTA, and Limitations

Pacheco et al. [14] show metadata fusion improves AUC by 3.1 pp. Groh et
al. [15] caution about skin-tone bias. TTA [17] averages predictions over
augmented views. Fujisawa et al. [18] identify amelanotic and small lesions
as persistent DL failure modes. Brancaccio et al. [19] note AUC overstates
real-world utility, motivating per-category error reporting (Section IV-G).

---

[← I. Introduction](02_introduction.md) | [Next → III. Methodology](04_methodology.md)
