# I. Introduction

Skin cancer is the most common malignancy worldwide, with melanoma accounting
for fewer than 5% of skin cancer cases yet more than 75% of skin cancer deaths
[1]. The American Cancer Society estimated over 100,000 new melanoma diagnoses
in the United States in 2020 alone. Five-year survival exceeds 98% when melanoma
is detected at stage I, but falls to below 25% for metastatic stage IV disease,
making early and accurate diagnosis one of the most clinically impactful
intervention points available to healthcare providers [2].

Dermoscopy—a non-invasive technique that uses polarised or cross-polarised light
to visualise sub-surface skin structures—has become the standard imaging modality
for lesion evaluation. Studies consistently show that dermoscopy improves
diagnostic sensitivity by 10–27 percentage points over naked-eye examination [3].
However, accurate interpretation of dermoscopic images requires years of
specialist training, and inter-observer variability among board-certified
dermatologists remains a well-documented clinical challenge. This variability,
combined with a global shortage of dermatologists—particularly in low- and
middle-income countries—has driven sustained interest in computer-aided diagnosis
(CAD) as a decision-support tool.

The landmark result of Esteva et al. [5], who showed a fine-tuned Inception-v3
network matched the accuracy of 21 board-certified dermatologists on a 2,032-
image test set, established deep learning as a clinically credible approach and
spurred the creation of the International Skin Imaging Collaboration (ISIC)
challenge series. Since 2016, successive ISIC challenges have expanded the scope
from binary classification to multi-class diagnosis, lesion segmentation, and,
most recently, patient-centric multi-lesion analysis [6, 7]. The ISIC 2020
challenge introduced per-image patient identifiers and structured metadata
(age, sex, anatomical site) alongside dermoscopic images, explicitly
acknowledging that dermatologists do not diagnose from single isolated images
but rather from a holistic clinical picture [7].

A persistent limitation of image-only classifiers is that they discard this
routinely available clinical context. Patient age, biological sex, and lesion
anatomical location each carry independent epidemiological weight: incidence
rates peak in older males, and torso lesions in men carry elevated risk relative
to other anatomical sites [4]. Multi-modal architectures that late-fuse CNN
image embeddings with tabular patient data offer a principled approach to
incorporating this prior into the learned decision boundary.

A further structural challenge is severe class imbalance. In the ISIC 2020
dataset, malignant samples represent only approximately 1.76% of images in the
raw challenge split; in the version used here—augmented with external malignant
images—the positive rate is ~13.6%, still presenting a significant 6.4:1
imbalance that standard cross-entropy loss handles poorly, tending to produce
high accuracy but low sensitivity classifiers.

This work makes the following contributions:
1. A reproducible dual-branch fusion model combining a pretrained CNN backbone
   with a metadata MLP, trained end-to-end under class-imbalanced conditions
   using sigmoid focal loss with a mathematically derived α=0.864.
2. A controlled comparison of three widely-used backbones—EfficientNet-B0,
   DenseNet-121, and ResNet-50—under identical hyperparameters, including
   inference time and convergence stability analysis.
3. Demonstration that the metadata branch improves recall by approximately 2.8%
   over an image-only baseline, consistent with findings in the literature [14].
4. An implemented test-time augmentation (TTA) pipeline (six geometric
   transforms) and analysis of its effect on F1 and recall stability.
5. Structured error analysis identifying systematic false-negative patterns
   (amelanotic melanomas, small lesions) and EigenCAM visualisations validating
   model attention against ABCDE dermatological criteria.

### Scope

**In scope:**
- Binary malignancy classification (benign vs. malignant) from 256×256
  dermoscopic images paired with three structured metadata fields (age, sex,
  anatomical site) from the SIIM-ISIC 2020 dataset.
- End-to-end training and validation of three CNN backbones (EfficientNet-B0,
  DenseNet-121, ResNet-50) with a fixed 20-epoch, single-GPU training budget.
- Late-fusion of image and metadata modalities via a dual-branch architecture.
- Class-imbalance handling via sigmoid focal loss with class-proportional
  alpha weighting (mathematically derived from class distribution).
- Test-time augmentation (horizontal flip and five additional geometric
  transforms) for conservative inference-time probability estimation.
- Evaluation using accuracy, recall (sensitivity), specificity, PPV, and F1
  on a held-out validation split; structured error analysis of FN/FP patterns.
- EigenCAM visual explainability with a Gradio inference application.

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

| | |
|---|---|
| [← Abstract](01_abstract.md) | [Next → II. Literature Review](03_literature_review.md) |
