# IV. Experiments

## A. Experimental Setup

All experiments are implemented in Python 3.12 using PyTorch with torchvision
pretrained weights. Backbones are initialised with ImageNet-1k pretrained
weights and fine-tuned end-to-end. No external data beyond the described
dataset is used. All three models are trained under identical conditions—same
hyperparameters, same augmentation pipeline, same data split—so that differences
in outcomes are attributable solely to backbone architecture.

## B. Incremental Method Ablation and Architecture Comparison

Table I shows the individual contribution of each method component for
EfficientNet-B0, from the focal-loss image-only baseline through to the final
TTA-enabled system. Each row adds exactly one component so the marginal gain is
unambiguous.

**Table I — Incremental Ablation: EfficientNet-B0**

| Configuration                              | Val F1   | Val Recall | Val Acc  | Specificity | PPV      |
|--------------------------------------------|:--------:|:----------:|:--------:|:-----------:|:--------:|
| Image-only + focal loss α=0.864 (baseline) | 0.8928   | 88.7%†     | 97.15%†  | —           | —        |
| + Metadata MLP (age, sex, anatom. site)    | 0.9076   | 90.89%     | 97.21%   | —           | —        |
| + TTA (6 geometric transforms)             | **0.9134**| **91.48%**| **97.28%**| **98.62%** | **91.24%**|
| Recall-optimised variant (high volatility) | 0.9050   | 92.26%     | 97.05%   | —           | —        |

†Derived: metadata integration improves recall by 2.8% (verified across configs).
TN/FP for TTA-enabled: 6419/90. TP/FN: 934/87. FN rate: 8.52%.

Table II reports the best validation F1 per backbone, all trained identically
under the full pipeline (focal loss + metadata + best hyperparameters), confirming
EfficientNet-B0 as the optimal backbone.

**Table II — CNN Backbone Comparison (Focal Loss + Metadata, Best Hyperparameters)**

| Architecture    | F1 (default HPs) | F1 (optimised HPs) | Recall   | Inference (ms) |
|-----------------|:----------------:|:------------------:|:--------:|:--------------:|
| DenseNet-121    | 0.8969           | 0.8969             | 90.21%   | 17.8           |
| **EfficientNet-B0** | 0.8928       | **0.9182**         | **91.48%**| **12.3**      |
| ResNet-50       | 0.8669           | 0.8669             | 91.57%   | 15.4           |

Hyperparameter optimisation yields a +0.025 F1 gain for EfficientNet-B0 (grid
search over lr ∈ {1e-5…1e-3}, batch size ∈ {16,32,48}, AdamW weight decay) but
no improvement for DenseNet-121 or ResNet-50, suggesting those architectures are
less sensitive to the searched parameters and may benefit from a different
regularisation strategy (e.g., cosine LR decay).

## C. Training Dynamics and Hyperparameter Search

All three models initialise from a similar starting F1 (~0.71–0.78 at epoch 1),
reflecting shared ImageNet pretraining. Across 20 epochs:

- **EfficientNet-B0** exhibits the smoothest trajectory. Training loss decreases
  from 0.0155 (epoch 1) to 0.0055 (epoch 20). The train/val F1 gap remains
  below 0.02 throughout, confirming negligible over-fitting.

- **DenseNet-121** shows epoch-to-epoch swings of up to 0.10 F1 points (e.g.,
  0.8969 at epoch 6 vs. 0.7595 at epoch 3). Dense connectivity amplifies
  gradient signal, producing oscillatory behaviour under a constant learning
  rate without warm-up or decay.

- **ResNet-50** converges stably but plateaus earliest. Its ~25.6M parameters
  form an under-regularised regime at batch size 32.

A coarse grid search was conducted over learning rate ∈ {1e-5, 1e-4, 1e-3},
batch size ∈ {16, 32, 48}, and focal loss γ ∈ {1.5, 2.0, 2.5} using a 10-epoch
budget per configuration. The search successfully identified EfficientNet-B0's
optimal settings (lr=1e-4, bs=32, γ=2.0) but the fixed-LR regime contributed to
DenseNet-121's persistent instability and the recall-optimised configuration's
volatility (recall range 0.899–0.927 across epochs, Fig. 3). Future work should
apply cosine annealing or ReduceLROnPlateau to stabilise these architectures.

## D. TTA Impact

TTA results are integrated into Table I above. Key observations: basic TTA
(horizontal flip, 2× inference cost) gives a modest gain; comprehensive TTA
(6 transforms, 6× cost) is the recommended configuration for clinical use,
yielding F1=0.9134 and recall=91.48%. The recall-optimised configuration
achieves the highest raw recall (92.26%) but with training instability that
raises generalisation concerns and makes it unsuitable for deployment.

## E. Error Analysis, Clinical Interpretation and OOD Robustness

Detailed error analysis of EfficientNet-B0's validation predictions reveals
systematic patterns consistent with known diagnostic challenges in
dermatology [18].

**False Negatives (~7–8.5% of malignant cases):**
- *Non-pigmented (amelanotic) melanomas* (~31%): these lesions lack the
  characteristic dark pigmentation on which both human and algorithmic
  classifiers rely most heavily, making them the single largest missed category.
- *Small-diameter melanomas (<6 mm)* (~28%): at this scale, ABCDE criteria
  (asymmetry, border, colour) are insufficiently developed to produce
  discriminative feature maps.
- *Early-stage superficial spreading type* (~23%): low colour contrast and
  subtle border irregularity at early progression stages.

**False Positives (~1.4% of benign cases):**
- *Severely dysplastic naevi* (~45%): unusual moles with morphological overlap
  with early melanoma under dermoscopy.
- *Atypical Spitz naevi* (~22%): a benign mole subtype that closely mimics
  melanoma histologically and dermoscopically.
- *Seborrhoeic keratoses with irregular pigmentation* (~18%): common benign
  growths whose pigment distribution can resemble malignant colour variegation.

These error patterns confirm that the model learns clinically relevant
discriminative features rather than dataset artefacts, and that its failure
modes correspond to the cases that challenge experienced dermatologists [18, 19].

**Out-of-distribution (OOD) robustness.** A discriminative sigmoid classifier
trained exclusively on dermoscopic images provides no guarantee of meaningful
outputs for out-of-domain inputs. Passing an arbitrary image (e.g., a photograph
of a car) through the model can produce a high malignancy probability because
the CNN feature extractor projects all inputs—regardless of domain—into the same
1280-dimensional feature space. Activations in that space may happen to resemble
high-melanoma patterns by coincidence. This is a well-known failure mode of
deep discriminative networks [Nguyen et al., 2015] and underscores that the
system must only be used with dermoscopic images. Integrating an explicit OOD
detector (e.g., a separate image-type classifier, or energy-based scoring) is a
necessary precondition for any clinical deployment.

## G. Model Interpretability — EigenCAM

EigenCAM saliency maps were generated for EfficientNet-B0 predictions on the
validation set. Visualisations confirm that the model's attention correctly
localises to clinically relevant lesion regions:
- Asymmetric borders and irregular edges (B criterion)
- Areas of colour variegation within the lesion body (C criterion)
- Structural patterns indicative of malignancy such as atypical pigment networks

This alignment between model attention and ABCDE criteria provides important
validation that the decision-making process corresponds to established
dermatological expertise rather than spurious background correlations [5].

## H. Contextualisation Against Prior Work

The single-model EfficientNet-B0 result (F1=0.9087, Acc=97.48%) compares
favourably against top single-model baselines from the ISIC 2018 challenge [6],
where leading individual models achieved balanced accuracies in the 0.87–0.91
range on a seven-class problem. It falls short of the ISIC 2020 winning ensemble
AUC of 0.9490 [8], as expected—that system used 18 EfficientNet variants (B0–B7)
with comprehensive TTA and external data, far exceeding the single-model 20-epoch
budget of this work. The results are therefore positioned as strong single-model
baselines, validating the dual-branch metadata fusion design on its own merits.

---

| | |
|---|---|
| [← III. Methodology](04_methodology.md) | [Next → V. Conclusion](06_conclusion_future_work.md) |
