# All 5 Experiments — The Full Chain

← [Focal Loss](03_focal_loss_imbalance.md) | [Index](index.md) | Next: [Optimizers →](05_optimizers_schedulers.md)

---

## Design Principle

Each experiment introduces **exactly one primary change** so every factor can be isolated. This is a proper ablation — not a random grid search.

**Fixed across all:** LR=10⁻⁴, batch=32, 20 epochs, same data split, same augmentation.

---

## The Full Table (Memorise This)

> **Column key:**  
> **F1** = harmonic mean of precision & recall — your primary optimisation target (higher = better balance of catching cancer without too many false alarms)  
> **Recall** = % of true melanomas caught — primary clinical metric (higher = fewer missed cancers)  
> **FN** = raw count of missed melanomas out of ~1,024 true malignant val cases (lower = fewer patients sent home with undetected cancer); Δ = change vs previous row  
> **FP** = raw count of benign cases wrongly flagged as malignant out of ~6,506 benign val cases (lower = fewer unnecessary biopsies)

| Config | Optim | Sched | γ | Meta | F1 (balance) | Recall (cancers caught) | FN Δ (missed cancers) | FP (false alarms) |
|--------|-------|-------|---|------|-------------|------------------------|----------------------|-------------------|
| Baseline | Adam | None | 2.0 | ✓ | 0.8932 | 92.16% | 80 (—) | 145 |
| Exp 1 (cosine) | Adam | Cosine | 2.0 | ✓ | **0.8984** | 93.14% | 70 (−10 fewer missed) | 145 (unchanged) |
| Exp 2 (AdamW) | AdamW | Cosine | 2.0 | ✓ | 0.8855 ↓ | **93.54%** ↑ | 66 (−4 fewer missed) | 181 (**+36 extra false alarms**) |
| Exp 3 (γ=1.5) | AdamW | Cosine | 1.5 | ✓ | 0.8968 | 92.75% | 74 (+8 more missed) | **144** (−37 fewer false alarms) |
| Exp 4 (no meta) | Adam | None | 2.0 | ✗ | 0.9000 | 90.30% ↓ | 99 (+19 more missed‡) | 106 (fewest false alarms) |

‡ vs Baseline, not previous row.

---

## Narrative: Reading the Chain

### Baseline → Exp 1 (Add Cosine Annealing)
**Change:** Learning rate scheduler (None → Cosine Annealing)  
**Result:** FN drops 80 → 70 (saves 10 patients). FP unchanged at 145. F1 rises 0.8932 → 0.8984.  
**Why it worked:** Cosine annealing lets the model settle into sharper, tighter minima — it can distinguish hard malignant features without accidentally widening the boundary for benign. This is the **single best change** in the whole study.

---

### Exp 1 → Exp 2 (Adam → AdamW + Weight Decay 10⁻⁵)
**Change:** Optimizer  
**Result:** FN drops further (70 → 66, best recall 93.54%). But FP surges 145 → 181 — 36 extra false alarms.  
**F1 drops** from 0.8984 → 0.8855 despite better recall. The increased FP swamp the precision term.  
**Why it failed:** Weight decay penalises large weights. For a standard vision task, this is good (prevents overfit). Here, the minority class *needs* sharper, larger activation boundaries to be distinguished. AdamW smoothed those boundaries, broadening the decision region and catching more melanoma but also flagging more benign lesions.

---

### Exp 2 → Exp 3 (γ 2.0 → 1.5)
**Change:** Focal loss γ  
**Result:** FP drops 181 → 144 (specificity rescued). FN rises 66 → 74 (some recall traded back).  
**F1 recovers** from 0.8855 → 0.8968.  
**Why it worked:** Lower γ means less aggressive suppression of easy benign examples. The network re-learns what clear benign cases look like, so it stops over-predicting malignant on ambiguous benign lesions. The cost is slightly weakened attention to hard malignant features.

---

### Baseline → Exp 4 (Remove Metadata Branch)
**Change:** Architecture (metadata branch disabled)  
**Result:** Accuracy *looks* better (97.28% vs ~96% for Baseline). F1 is 0.9000. But FN jumps to 99 — **19 more missed cancers** vs Baseline's 80.  
**Why this exposes the accuracy trap:** At 13.6% prevalence, 19 extra misses out of 7,530 samples barely moves accuracy. The all-benign predictor is 86.4% accurate. Accuracy is useless here.  
**Metadata gains:** +1.9 pp recall (99→80 FN) at the cost of 39 extra FP (106→145). The MLP learns to flag anatomical site + age combinations that correlate with melanoma risk, even when the image is ambiguous.

---

## Grilling Question: "If Exp 1 is your best model, why did you run Exp 2 and 3?"

**Because science requires falsification.** Exp 2 tests whether weight decay helps. It doesn't — it harms specificity. Exp 3 tests whether we can *recover* the specificity loss from Exp 2 by relaxing γ. It partially does, but not fully — F1=0.8968 doesn't beat Exp 1's 0.8984. This proves the Exp 1 configuration is genuinely optimal, not just accidentally chosen.

---

## Grilling Question: "All five experiments hit AUC=0.99. What does that tell you?"

AUC measures **ranking ability** — whether the model assigns higher scores to true positives than true negatives, across all possible thresholds. AUC=0.99 across all experiments means the HP changes do **not** affect the fundamental discriminative power of the network.

What changes is the **operating point** — where you draw the line between "malignant" and "benign." Cosine annealing, AdamW, γ — these shift the precision-recall tradeoff at the chosen threshold (0.5), not the underlying AUC. This is consistent with Davis & Goadrich (2006): AUC can be misleading on imbalanced data precisely because it averages over all thresholds, including clinically irrelevant ones.

---

## Grilling Question: "How did you select the best checkpoint?"

By **validation F1**, not validation recall. 

- A model that predicts malignant for every sample has recall=100% and is clinically useless.
- F1 = $\frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$ — the harmonic mean penalises any imbalance between precision and recall.
- It forces the model to balance catching cancer (recall) against not flooding clinics with false alarms (precision).

---

## Layman Analogy

*"The five experiments are like five cooking trials for a recipe, changing one ingredient each time. Exp 1 changes the oven temperature setting (cosine annealing) — the cake rises perfectly. Exp 2 changes the flour type (AdamW) — it's too moist, falls apart. Exp 3 reduces the sugar (γ=1.5) to compensate — better, but not as good as Exp 1. Exp 4 removes the secret sauce entirely (no metadata) — the cake looks fine but tastes wrong. Exp 1 wins, and now you can prove *exactly* why."*

---

← [Focal Loss](03_focal_loss_imbalance.md) | [Index](index.md) | Next: [Optimizers →](05_optimizers_schedulers.md)
