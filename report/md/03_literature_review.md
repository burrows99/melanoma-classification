# II. Literature Review

## A. ISIC Benchmarks and CNN Backbones

The ISIC challenge series has driven algorithmic progress in dermoscopy since
2016 [6]. The 2020 SIIM-ISIC challenge [7] represented a conceptual shift toward
patient-centric diagnosis: 33,126 images from 2,056 patients with
histopathologically confirmed labels were released alongside structured metadata
(age, sex, anatomical site), reflecting the clinical reality that dermatologists
integrate patient history when diagnosing. The winning submission [8] achieved
AUC=0.9490 via an ensemble of EfficientNet variants (B0–B7) with comprehensive
TTA, establishing EfficientNet as the backbone of choice for this task.

Three backbones are evaluated in this work. **EfficientNet** [9] applies
compound scaling of depth, width, and resolution; its base model (B0, 5.3M
parameters) achieves strong ImageNet performance with minimal FLOPs and
excellent transfer learning properties. **DenseNet-121** [10] employs dense
connectivity to promote feature reuse; Rajpurkar et al. [11] demonstrated
radiologist-level pathology detection with this architecture, validating dense
connectivity as an effective inductive bias for medical imaging. **ResNet-50**
[12] uses residual skip connections for stable deep training and remains the
canonical medical imaging baseline.

## B. Class Imbalance

At ~13.6% positive rate, a model predicting benign for all inputs achieves
86.4% accuracy—a degenerate local minimum for cross-entropy training. Focal
loss [13] modulates the standard cross-entropy by $(1-p_t)^\gamma$,
concentrating gradient on hard cases and setting $\alpha$ inversely proportional
to class frequency. Multiple top-10 ISIC 2020 submissions employed focal loss
[7]. Setting α = 1 − 0.136 = 0.864 is a principled derivation from class
statistics, rather than a tuned hyperparameter.

## C. Metadata Fusion, TTA, and Known Limitations

Dermatologists integrate clinical context when diagnosing; Pacheco et al. [14]
show that appending age, sex, and lesion site to a MobileNet embedding improved
AUC by 3.1 pp on PAD-UFES-20. Zhang et al. [16] confirm that metadata fusion
combined with self-supervised pretraining yields consistent AUC gains. Groh et
al. [15] caution that models trained on light-skinned cohorts generalise poorly
to diverse skin tones, motivating transparent demographic stratification.
TTA reduces prediction variance by averaging predictions over multiple
augmented views [17]; both a basic (h-flip) and comprehensive (6-transform)
strategy are implemented here, consistent with practice in top ISIC submissions
[8]. Fujisawa et al. [18] identify amelanotic melanomas and small lesions
(<6 mm) as consistent DL failure modes—the same categories dominating the false
negatives in this work. Brancaccio et al. [19] note that published AUCs often
overstate real-world utility due to selection bias, motivating per-category
error reporting (Section IV-E).

---

[← I. Introduction](02_introduction.md) | [Next → III. Methodology](04_methodology.md)
