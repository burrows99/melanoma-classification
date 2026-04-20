# Evaluation Metrics & Test-Time Augmentation

← [Optimizers](05_optimizers_schedulers.md) | [Index](index.md) | Next: [Metadata & SHAP →](07_metadata_shap.md)

---

## The Five Metrics

$$\text{Recall (Sensitivity)} = \frac{TP}{TP + FN}$$

$$\text{Specificity} = \frac{TN}{TN + FP}$$

$$\text{PPV (Precision)} = \frac{TP}{TP + FP}$$

$$\text{F1} = \frac{2 \cdot \text{PPV} \cdot \text{Recall}}{\text{PPV} + \text{Recall}} = \frac{2TP}{2TP + FP + FN}$$

$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

---

## Grilling Question 1: "Why is recall the primary metric?"

**Clinical asymmetry of errors:**

| Error | What happens | Clinical consequence |
|-------|-------------|---------------------|
| False Negative (missed cancer) | Malignant predicted as benign | Patient goes untreated → metastasis → <25% 5-yr survival |
| False Positive (false alarm) | Benign predicted as malignant | Unnecessary biopsy → anxiety, cost, minor procedure |

Missing a melanoma is **not recoverable** — once it metastasises, survival drops from >98% to <25%. A false alarm causes a biopsy; uncomfortable but not fatal. Therefore, maximising recall (minimising FN) is the primary objective.

**Drill-Down:** *Why not maximise recall completely — just predict everyone as malignant?*  
A model predicting 100% malignant has Recall=1.0 and is clinically useless — every patient gets unnecessary surgery. F1 prevents this: Precision would be 0.136 (only 13.6% of flagged cases are truly malignant), dragging F1 to 0.24. Checkpoint selection by val F1 enforces this balance.

---

## Grilling Question 2: "Why is accuracy useless here?"

At 13.6% malignant prevalence, an **all-benign predictor** achieves:
$$\text{Accuracy} = \frac{0 + 6490}{0 + 6490 + 1040 + 0} = 86.4\%$$

It catches **zero cancers**. Yet 86.4% accuracy sounds impressive. This is the *accuracy paradox* on imbalanced datasets.

Exp 4 (image-only) has 97.28% accuracy but 99 FN (19 more missed cancers than Baseline). Its "better" accuracy reflects fewer FP (106 vs 145) — because without the metadata MLP being "paranoid" about anatomical site, it's less trigger-happy. But fewer FP at the cost of more FN is clinically wrong.

---

## Grilling Question 3: "What is AUC and why does it reach 0.99 across all experiments?"

**AUC (Area Under the ROC Curve)** measures the probability that a randomly chosen malignant case scores *higher* than a randomly chosen benign case, across all possible thresholds.

All five experiments achieve AUC=0.99. This means the underlying feature representation of the model is excellent at *ranking* malignant above benign — the backbone has learned discriminative features regardless of which HP configuration was used.

HP changes (scheduler, optimizer, γ) only shift the **operating point** — i.e., which threshold is chosen to call "positive." AUC aggregates over all thresholds, smoothing out these differences.

**Why AUC overstates clinical utility (Davis & Goadrich, 2006):** On imbalanced data, ROC curves include operating points at very low FPR (many thresholds where almost nothing is flagged). These inflate AUC without reflecting real clinical utility. The Precision-Recall curve is more honest: it directly shows the Recall vs FP tradeoff that matters clinically.

---

## Grilling Question 4: "Explain your Test-Time Augmentation. How does it work mathematically?"

**TTA:** Instead of one forward pass per image, run N augmented versions and average the predictions.

**2-transform TTA (evaluation pipeline):**
$$\hat{p} = \frac{1}{2}\left(f(\mathbf{x}) + f(\text{hflip}(\mathbf{x}))\right)$$

**6-transform TTA (Gradio demo):**
$$\hat{p} = \frac{1}{6}\sum_{k=1}^{6} f(\mathbf{x}_k)$$

where $\mathbf{x}_k \in \{\text{original, h-flip, v-flip, 90°, 180°, 270°}\}$.

**Why does averaging reduce variance?** Each prediction is $\hat{p}_k = p + \epsilon_k$ where $p$ is the true probability and $\epsilon_k$ is orientation-specific noise. The average:
$$\mathbb{E}\left[\frac{1}{N}\sum_k \hat{p}_k\right] = p, \quad \text{Var}\left[\frac{1}{N}\sum_k \hat{p}_k\right] = \frac{\sigma^2}{N}$$

Variance scales as 1/N — 6 transforms reduces prediction variance to 1/6 of a single-pass estimate.

**Why dermoscopic images specifically?** Melanoma has no canonical orientation — a lesion doesn't "face" a particular direction. Horizontal flip and 90° rotations are semantically equivalent. A model that predicts 0.7 on the original but 0.4 on the h-flipped version has high orientation variance; TTA averages these to a stable 0.55.

---

## Grilling Question 5: "What happened to metrics when you applied 2-transform TTA to the Baseline?"

| | Recall | Specificity | F1 | FN | FP |
|---|--------|------------|----|----|-----|
| Baseline (no TTA) | 92.16% | — | 0.8932 | 80 | 145 |
| Baseline + 2-TTA | 91.48% | **98.62%** | **0.9134** | 87 | **90** |

TTA raised F1 by +0.0202. The averaging tightened the decision boundary — borderline benign cases that one augmentation called malignant got averaged down. This traded 7 extra FN for 55 fewer FP. In a **screening** context, this tradeoff might be acceptable (fewer unnecessary biopsies). In a **high-sensitivity** context, you'd prefer Exp 1 without TTA.

---

## Layman Analogy

*"TTA is like asking three colleagues to proofread the same document independently, then only flagging an error if the majority agree. One tired proofreader might miss things or flag non-errors. The majority vote is more reliable.*

*Accuracy on imbalanced data is like judging a fire alarm system by how quiet it is. A system that never triggers gets 100% 'peace and quiet' — but when there's a fire, you're dead. You need to measure how often it catches fires (recall) and how often it's a false alarm (1-specificity)."*

---

← [Optimizers](05_optimizers_schedulers.md) | [Index](index.md) | Next: [Metadata & SHAP →](07_metadata_shap.md)
