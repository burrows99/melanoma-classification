# Viva Prep — Master Index & Quick-Fire Overview

> **Strategy:** Each file follows: Grilling Question → Core Reasoning → The Tradeoff → Drill-Down Q&A → Layman Analogy.  
> Read files in order for a full narrative, or jump to the section you're weakest on.

---

## Files

| # | Topic | Key Exam Threat |
|---|-------|-----------------|
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
