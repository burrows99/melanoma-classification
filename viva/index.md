# Viva Prep — Master Index & Quick-Fire Overview

> **Strategy:** Each file follows: Grilling Question → Core Reasoning → The Tradeoff → Drill-Down Q&A → Layman Analogy.  
> Read files in order for a full narrative, or jump to the section you're weakest on.

---

## Files

| # | Topic | Key Exam Threat |
|---|-------|-----------------|
| [basic.md](basic.md) | **Detailed Chapters 1–4 (original notes)** | Deep narrative with layman analogies |
| [01](01_dataset_preprocessing.md) | **Dataset & Preprocessing** | Why stratified split? What does CLAHE do? |
| [02](02_architecture.md) | **Model Architecture** | Why late fusion? What is 1344-d? |
| [03](03_focal_loss_imbalance.md) | **Focal Loss & Class Imbalance** | Derive α. What does γ suppress? |
| [04](04_experiments.md) | **All 5 Experiments — The Full Chain** | Reproduce Table I from memory |
| [05](05_optimizers_schedulers.md) | **Optimizers & Schedulers** | AdamW vs Adam, cosine annealing math |
| [06](06_evaluation_metrics.md) | **Evaluation Metrics & TTA** | Why not accuracy? TTA variance reduction |
| [07](07_metadata_shap.md) | **Metadata Fusion & SHAP** | Prove metadata isn't noise |
| [08](08_ood_detection.md) | **OOD Detection (Mahalanobis)** | Euclidean vs Mahalanobis, threshold math |
| [09](09_eigencam_interpretability.md) | **EigenCAM & Interpretability** | ABCDE criteria alignment |
| [10](10_error_analysis.md) | **Error Analysis — FN & FP** | Clinical implications of each error type |
| [11](11_literature_context.md) | **Literature & Design Choices** | Why not ViT? Why not ResNet-50? |

---

## The 5 Numbers (Say These In Your Sleep)

| Number | Meaning |
|--------|---------|
| **13.6%** | Malignant prevalence → all-benign predictor = 86.4% acc |
| **5.3M** | EfficientNet-B0 parameters |
| **1344-d** | Fused embedding: 1280 (image) + 64 (metadata). EfficientNet-B0 global pool → 1280-d; MLP 14→128→64-d; concatenated before final Linear → sigmoid |
| **α = 0.864** | Focal loss alpha = 1 − 0.136 |
| **F1 = 0.9134** | Best result: Baseline + 2-transform TTA |

---

## The Single Most Important Table (Memorise This)

| Config | Optim | Sched | γ | Meta | F1 | Recall | FN | FP |
|--------|-------|-------|---|------|----|--------|----|----|
| Baseline | Adam | None | 2.0 | ✓ | 0.8932 | 92.16% | 80 | 145 |
| Exp 1 (cosine) | Adam | Cosine | 2.0 | ✓ | **0.8984** | 93.14% | **70** | 145 |
| Exp 2 (AdamW) | AdamW | Cosine | 2.0 | ✓ | 0.8855 | **93.54%** | **66** | 181 |
| Exp 3 (γ=1.5) | AdamW | Cosine | 1.5 | ✓ | 0.8968 | 92.75% | 74 | **144** |
| Exp 4 (no meta) | Adam | None | 2.0 | ✗ | 0.9000 | 90.30% | 99 | 106 |

**Key narrative:** Cosine annealing → best single change. AdamW → highest recall but FP catastrophe. γ=1.5 → specificity rescue. Metadata → +1.9pp recall, +39 FP.

---

## Chapter Summaries (5-Minute Speed-Read)

### Chapter 1 — Backbone & Loss

- **Why EfficientNet-B0?** Compound scaling (depth × width × resolution jointly). 5.3M params, 77.1% ImageNet. ISIC 2020 winning solution used EfficientNet. Best accuracy-per-FLOP.
- **Why not ResNet-50?** 25.6M params, 76.1% acc — more expensive, less accurate.
- **Why focal loss?** Standard cross-entropy lets the model ignore 86% benign and still minimise loss. Focal loss: $FL(p_t) = -\alpha_t(1-p_t)^\gamma \log(p_t)$.
- **α = 0.864** = inverse frequency weight. Forces 6.4× more gradient attention per malignant sample.
- **γ = 2.0** suppresses easy benign examples. Too high → FP spike (model forgets benign). Too low → recall drops.
- **Exp 3 proof:** γ 2.0→1.5 dropped FP 181→144 but raised FN 66→74. Exactly as theory predicts.

### Chapter 2 — Architecture

- **Dual-branch late fusion:** CNN (1280-d) + MLP (14→128→64-d) → concatenate (1344-d) → Linear → sigmoid.
- **Why late?** Image and tabular data need separate "languages." Early fusion = meaningless (age ≠ pixel value). Late fusion mirrors clinical practice: scan first, patient chart at decision time.
- **Metadata preprocessing:** age → median impute + z-score; sex + anatom_site → `'Unknown'` fill + one-hot → 14-d vector.
- **MLP:** BatchNorm + ReLU + Dropout(0.3) after each layer. Dropout prevents memorisation of demographic patterns.
- **Exp 4 (no meta):** accuracy=97.28% (looks great), but FN=99 vs Baseline's 80. **19 more dead patients.** Accuracy is a trap.
- **Metadata adds:** +1.9pp recall (99→80 FN) at cost of +39 FP (106→145). Net gain is real.

### Chapter 3 — Optimizers & Schedulers

- **Adam:** adaptive learning rates per parameter. Good for sparse/noisy gradients from imbalanced data.
- **Cosine annealing:** $\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max}-\eta_{\min})(1+\cos\frac{t\pi}{T})$. Smooth LR decay → settles into sharp minima.
- **Exp 1 (cosine):** FN 80→70. Zero extra FP. Best single change. ΔF1=+0.0052.
- **AdamW:** decoupled weight decay applied directly to weights (not folded into gradient). Penalises large weights uniformly.
- **Exp 2 (AdamW):** Best recall (93.54%, FN=66) BUT FP surged 145→181. Weight decay eroded sharp minority-class boundaries.
- **Exp 3 (γ=1.5):** Partial recovery — FP back to 144, FN rises to 74. F1=0.8968.
- **Checkpoint selection:** validation F1, not recall. Perfect recall = predict everything malignant = clinically useless.

### Chapter 4 — Evaluation & TTA

- **Primary metric: Recall.** Missed melanoma = metastasis = <25% 5yr survival. False alarm = biopsy = uncomfortable but safe.
- **Accuracy trap:** 86.4% for all-benign. Always quote confusion matrix numbers.
- **AUC=0.99 across all experiments:** HP changes don't affect *ranking* ability, only the operating *threshold*. AUC overstates utility on imbalanced data (Davis & Goadrich 2006).
- **2-transform TTA (Baseline):** original + h-flip → average. Recall 92.16%→91.48%, but F1 0.8932→**0.9134**, specificity **98.62%**. Trades 7 FN for 55 fewer FP.
- **Why averaging reduces variance:** $\text{Var}[\bar{p}] = \sigma^2/N$. Orientation noise averages out.
- **6-transform TTA (Gradio demo):** + v-flip, 90°, 180°, 270°. Dermoscopy has no canonical orientation.

### Chapter 5 — Metadata & SHAP

- **SHAP (KernelExplainer):** model-agnostic, treats full model as black box. Computes Shapley values = marginal contribution of each feature averaged over all possible insertion orders.
- **Top SHAP features (Exp 1):** `anatom_site_torso` > `age_approx` (monotonic: older→higher risk) > `anatom_site_anterior_torso` (rare but SHAP up to 0.18) > sex (modest).
- **Why torso is dominant:** high UV exposure, large flat lesions often missed in self-exam, over-represented in confirmed melanoma in SIIM-ISIC.
- **Exp 3 exception:** γ=1.5 compresses SHAP magnitudes and distributes importance more evenly across features.
- **Three-pronged proof metadata isn't noise:** (1) Exp 4 ablation, (2) non-zero SHAP values, (3) clinical coherence with known epidemiology.

### Chapter 6 — OOD Detection

- **Problem:** Sigmoid always outputs a number. Koala → 85.8% malignant. No error thrown.
- **Solution:** Mahalanobis distance on the 1280-d backbone embedding *before* the logit.

$$D_M(\mathbf{z}) = (\mathbf{z}-\mu)^T \Sigma^{-1} (\mathbf{z}-\mu)$$

> Code uses **squared** Mahalanobis (no sqrt) — threshold is also squared, so the OOD check is internally consistent.

- **Reference distribution:** compute $\mu$ and $\Sigma^{-1}$ on all validation embeddings.
- **Threshold:** $\mu_{D_M} + 3\sigma_{D_M}$ → captures 99.7% of valid distribution.
- **Koala numbers:** distance=5124, threshold (Exp 1)=2652. Nearly 2× the threshold. Intercepted.
- **Why not Euclidean?** Assumes all 1280 dims equally scaled and independent. CNN features are correlated. Mahalanobis uses $\Sigma^{-1}$ to decorrelate and normalise.
- **Thresholds by experiment:** Exp 3=2637 (tightest), Baseline=3248 (loosest). Cosine-annealed models → tighter feature distributions.

### Chapter 7 — EigenCAM

- **What it shows:** heatmap of where EfficientNet-B0 is most activated in the final conv layer.
- **Method:** SVD of final feature map tensor → first right singular vector → reshape to $H \times W$.
- **Result (Exp 1):** attention on asymmetric borders, colour variegation, atypical pigment networks → matches ABCDE criteria.
- **Why not Grad-CAM?** Needs gradients through fusion head. EigenCAM is gradient-free, stable, faster.
- **Limitation:** not class-discriminative. Shows activation intensity, not specifically "what drove malignant vs benign."

### Chapter 8 — Error Analysis

- **FN types (6.8–8.5% of malignant):** amelanotic melanoma (no pigment → CNN can't grip), small lesions (<6mm at 256px resolution), early superficial spreading (subtle borders).
- **FP types (1.4–2.8% of benign):** dysplastic naevi, atypical Spitz naevi, irregular seborrhoeic keratoses. These are the *same* hard cases a dermatologist would biopsy — FP is defensible.
- **Best FN count:** Exp 2 (66) — but FP=181.
- **Best FP count with metadata:** Exp 3 (144) — but FN=74.
- **Best balanced:** Exp 1 (FN=70, FP=145, F1=0.8984).
- **6.86% FN rate** is competitive — human dermoscopy without AI has 10–20% miss rate.

---

## Literature Flashcards

| Paper | One-liner |
|-------|-----------|
| Esteva et al., *Nature* 2017 | CNN matched 21 dermatologists |
| Rotemberg et al., *Sci. Data* 2021 | ISIC 2020 dataset with metadata |
| Ha et al., arXiv 2020 | EfficientNet ensemble won ISIC 2020, AUC=0.949 |
| Tan & Le, *ICML* 2019 | EfficientNet compound scaling |
| Lin et al., *ICCV* 2017 | Focal loss — designed for object detection class imbalance |
| Fujisawa et al., *Br. J. Dermatol.* 2019 | Amelanotic + small lesions = persistent DL failure |
| Davis & Goadrich, *ICML* 2006 | AUC overstates utility on imbalanced data |
| Lee et al., *NeurIPS* 2018 | Mahalanobis OOD detection framework |

---

## The "Why Not?" Quick-Fire

| "Why not X?" | One-line answer |
|-------------|----------------|
| ViT | No translation equivariance → needs 10–100× more data at this scale |
| ResNet-50 | 4.8× more params, lower ImageNet accuracy |
| Vertical flip in val | Val uses resize+normalise only; v-flip is training-only |
| Oversampling | Memorises duplicate malignant images |
| Undersampling | Discards 86% of training data |
| Early fusion | Age has no spatial meaning in a pixel grid |
| Softmax | Creates class competition; imbalanced class wins by default |
| Euclidean OOD | Ignores feature correlations and scale differences |
| Val Recall for checkpoint | 100% recall model predicts everything malignant — useless |
| Accuracy as primary metric | All-benign model hits 86.4% and catches zero cancers |
