# II. Literature Review

## A. The ISIC Challenge Series and Benchmark Evolution

The International Skin Imaging Collaboration (ISIC) has organised the world's
largest public dermoscopy benchmark series since 2016, providing standardised
datasets and evaluation protocols that have shaped algorithmic progress in the
field. The 2018 challenge [6] attracted 159 submissions to its disease
classification task across 12,500 images and seven diagnostic categories,
establishing multi-class skin lesion classification as a tractable deep learning
problem. A key finding was that algorithms with equal aggregate test performance
could exhibit substantially different generalisation profiles—a critical warning
for clinical deployment.

The 2020 SIIM-ISIC challenge [7] represented a conceptual shift toward
patient-centric diagnosis. The dataset comprises 33,126 dermoscopic images
from 2,056 patients (mean 16 lesions per patient) with 584 histopathologically
confirmed melanomas. Crucially, per-image patient identifiers and structured
metadata fields—age, sex, and anatomical site—were released alongside images,
reflecting the clinical reality that dermatologists integrate patient history
when forming diagnoses. The winning submission by Ha et al. [8] achieved an
AUC of 0.9490 on the private leaderboard through an ensemble of EfficientNet
models at multiple scales (B0 to B7), with a subset of models incorporating
patient-level metadata. Their results established EfficientNet variants as the
architecture of choice for this task.

## B. CNN Backbone Architectures for Medical Imaging

**EfficientNet** [9], introduced by Tan and Le, applies compound scaling derived
from a neural architecture search to simultaneously adjust network depth, width,
and input resolution according to a fixed coefficient. EfficientNet-B0, the base
model with 5.3M parameters, achieves a top-1 accuracy of 77.1% on ImageNet with
significantly fewer FLOPs than comparable architectures. Its compact footprint
and strong transfer learning properties have made it the dominant backbone in
recent ISIC submissions. The Ha et al. winning solution [8] used no architecture
other than EfficientNet variants, reinforcing its suitability for dermoscopy.

**DenseNet** [10], proposed by Huang et al., employs dense block connectivity
in which each layer receives concatenated feature maps from all preceding layers
in the same block. This design promotes feature reuse, substantially reduces the
number of parameters relative to equivalent-depth VGG-style networks, and
alleviates vanishing gradients. Rajpurkar et al. [11] demonstrated DenseNet-121's
capability in medical imaging by training a 121-layer model to detect 14 chest
X-ray pathologies at radiologist-level performance, validating dense connectivity
as a strong inductive bias for fine-grained medical classification.

**ResNet** [12], introduced by He et al., uses residual skip connections that
allow gradient flow to bypass stacked convolutional layers, enabling training of
very deep networks. ResNet-50 remains the canonical baseline across virtually
all medical imaging benchmarks due to its well-understood convergence behaviour,
extensive pretrained weight availability, and robust transfer learning record.

## C. Handling Class Imbalance in Clinical Datasets

Class imbalance is a structural feature of melanoma datasets: even the augmented
ISIC 2020 version used in this work contains only ~13.6% malignant samples.
With standard binary cross-entropy, a model predicting benign for all inputs
achieves 86.4% accuracy, creating a degenerate but numerically attractive local
minimum. Common mitigation strategies include oversampling minority examples
(SMOTE and its variants), undersampling the majority class, class-weighted loss
functions, and focal loss.

Lin et al. [13] introduced focal loss in the context of dense object detection
(RetinaNet) to address the extreme foreground/background imbalance in one-stage
detectors. The loss modulates the standard cross-entropy by a factor
$(1-p_t)^\gamma$, down-weighting the contribution of easy, well-classified
examples and concentrating gradient signal on hard or ambiguous cases. An
additional class-balancing parameter $\alpha$ can be set inversely proportional
to class frequency. Focal loss has since been widely adopted for imbalanced
medical classification problems; multiple top-10 submissions in the ISIC 2020
challenge employed it explicitly [7].

## D. Multi-modal Metadata Fusion

The clinical diagnosis of melanoma is inherently multi-modal: dermatologists
simultaneously evaluate dermoscopic appearance, lesion evolution, and patient
demographics. Rotemberg et al. [7] explicitly motivated the inclusion of metadata
in ISIC 2020 by noting that patient-level context is especially useful for ruling
out false positives in patients with many atypical nevi. Pacheco et al. [14]
demonstrate on the PAD-UFES-20 dataset—comprising 2,298 smartphone images with
associated patient records—that appending age, sex, and lesion site features to
a MobileNet image embedding as a late-fusion vector improved AUC by 3.1
percentage points over the image-only baseline.

Groh et al. [15] provide a cautionary perspective: deep networks trained
primarily on light-skinned cohorts (as in most ISIC releases) show significantly
worse performance on darker skin types, emphasising that metadata and dataset
demographics must be carefully monitored to avoid algorithmic bias in deployment.
This consideration motivates the explicit inclusion of patient metadata as a
learned input rather than a post-hoc correction.

The dominant fusion paradigm in high-performing ISIC 2020 systems is late fusion:
image and metadata branches are processed independently and their embeddings are
concatenated before the final classification head. This design preserves
modularity, simplifies debugging, and allows the network to learn the relative
weighting of each modality end-to-end. It is the approach adopted in this work.
Zhang et al. [16] further demonstrate that combining metadata integration with
self-supervised pretraining on unlabelled dermoscopic images yields consistent
AUC gains over supervised image-only baselines, reinforcing that patient-level
context remains a complementary signal even as backbone representations improve.

## E. Test-Time Augmentation in Medical Imaging

Test-time augmentation (TTA) reduces prediction variance at inference by
averaging model outputs across multiple transformed versions of the same input.
Minaee et al. [17] demonstrate that TTA substantially improves consistency in
deep learning-based medical image segmentation, particularly for small or
ambiguous structures—a finding directly applicable to small and amelanotic
melanomas that form a disproportionate share of false negatives in dermoscopy
classifiers. Two TTA strategies are common in the melanoma literature: a *basic*
variant that averages the original and horizontally-flipped prediction, and a
*comprehensive* variant that averages predictions across six geometric transforms
(original, horizontal flip, vertical flip, 90°, 180°, and 270° rotations). The
Ha et al. winning ISIC 2020 solution [8] used comprehensive TTA as standard
practice for all ensemble members. Both strategies are implemented in the present
work.

## F. Limitations of Deep Learning for Melanoma Diagnosis

Despite strong benchmark performance, clinical deployment of DL melanoma
classifiers faces several well-documented limitations. Fujisawa et al. [18]
identify three key failure modes: (a) poor performance on amelanotic melanomas
lacking characteristic pigmentation; (b) reduced sensitivity on small lesions
(<6 mm) where ABCDE criteria are insufficiently developed; and (c) confounding
with morphologically similar benign lesions such as severely dysplastic naevi
and atypical Spitz tumours. These same categories account for the majority of
false negatives and false positives observed in this work, validating that the
system faces clinically realistic rather than trivially solvable challenges.

Brancaccio et al. [19] provide a broader reality check on AI in skin cancer
diagnosis, noting that reported benchmark AUCs often overstate real-world
utility due to selection bias in test sets, distribution shift between
dermoscopy devices, and absence of lesion evolution data. These limitations
motivate transparent reporting of per-error-category breakdowns rather than
aggregate metrics alone, which is the approach taken in Section IV-E.

---

| | |
|---|---|
| [← I. Introduction](02_introduction.md) | [Next → III. Methodology](04_methodology.md) |
