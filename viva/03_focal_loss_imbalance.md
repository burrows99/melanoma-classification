# Focal Loss & Class Imbalance

← [Architecture](02_architecture.md) | [Index](index.md) | Next: [Experiments →](04_experiments.md)

---

## The Core Problem

13.6% of samples are malignant. A model that **always predicts "benign"** achieves **86.4% accuracy** while catching **zero cancers**. Standard cross-entropy fails because it optimises for the majority class.

---

## The Mathematics

### Standard Binary Cross-Entropy

$$\text{CE}(p_t) = -\log(p_t)$$

where $p_t = p$ if $y=1$ (malignant), else $p_t = 1-p$.

**Problem:** For a sample the model confidently gets right ($p_t = 0.95$), $-\log(0.95) \approx 0.05$ — tiny gradient. The model is overwhelmed by 86% easy benign examples, each contributing small losses that dominate the gradient update.

---

### Focal Loss (Lin et al., 2017)

$$FL(p_t) = -\alpha_t \cdot (1 - p_t)^\gamma \cdot \log(p_t)$$

**Two modifications:**

**1. α-weighting** — upweights the minority class:
$$\alpha_t = \alpha \text{ if } y=1, \quad \alpha_t = 1-\alpha \text{ if } y=0$$

Setting $\alpha = 1 - 0.136 = \mathbf{0.864}$: the malignant class (y=1) gets weight 0.864, benign gets weight 0.136. This forces the gradient to care **6.4× more** about each malignant example.

*Why 1 − prevalence?* It's the inverse-frequency weighting. If malignant is 13.6% of data, its examples should each carry proportionally more weight so the total gradient contribution is balanced.

**2. γ-modulation** — suppresses easy examples:

| $p_t$ | $(1-p_t)^2$ (γ=2) |
|--------|-------------------|
| 0.9 (easy correct) | 0.01 — almost no gradient |
| 0.5 (uncertain) | 0.25 — moderate gradient |
| 0.1 (wrong) | 0.81 — near-full gradient |

The model learns to **ignore the easy 86% benign examples** and concentrate gradients on hard, ambiguous cases — exactly where malignant features are contested.

---

## Grilling Question 1: "How did you derive α=0.864?"

**Directly from class prevalence:**  
$$\alpha = 1 - P(\text{malignant}) = 1 - 0.136 = 0.864$$

This is not a tuned hyperparameter — it's the mathematically principled inverse-frequency weight. It ensures that the sum of gradients from the minority class matches what it would be at 50/50 balance.

---

## Grilling Question 2: "What does γ actually do to the loss surface?"

γ controls the **rate at which easy examples are down-weighted**:

- **γ = 0:** Focal loss collapses to weighted cross-entropy (no suppression)
- **γ = 1:** Mild suppression; easy negatives still contribute some gradient
- **γ = 2:** (Baseline, Exp 1–2) — strong suppression; confident correct predictions contribute ~1% of their normal gradient
- **γ = 1.5:** (Exp 3) — moderate suppression; restores gradient to easy benign examples → fewer FP

**The tradeoff:** Higher γ → better recall (catches more melanoma) but risks forgetting what "easy benign" looks like → FP spike. Lower γ → better specificity but may miss hard malignant cases → FN rise.

**Experiment 3 proof:** Dropping γ from 2.0 → 1.5 dropped FP from 181 → 144 (+specificity) but raised FN from 66 → 74 (−recall). Exactly as theory predicts.

---

## Grilling Question 3: "Why sigmoid, not softmax?"

Softmax requires the probabilities across all classes to sum to 1 — it creates competition between "malignant" and "benign." With severe imbalance, the benign class wins this competition by default.

**Sigmoid** maps the raw logit to [0,1] independently for each class. The decision threshold (default 0.5) can be tuned without the classes competing. This is also why it's called **sigmoid** focal loss — one sigmoid output per sample, not a softmax over 2 classes.

---

## Grilling Question 4: "Why not simply oversample or undersample?"

| Approach | Problem |
|----------|---------|
| Random oversampling (duplicate malignant) | Model memorises exact malignant images — over-fits |
| Random undersampling (drop benign) | Throws away 86% of training data — wastes signal |
| SMOTE | Generates synthetic images in feature space — may not be realistic for dermoscopy |
| **Focal Loss** | No data modification; reweights gradients mathematically; uses all data |

Focal loss is the *principled* approach: it solves the imbalance problem at the gradient level, not the data level.

---

## Layman Analogy

*"Imagine you're training a student to identify rare antique coins (malignant) among a pile of common modern coins (benign). Standard cross-entropy is a teacher who marks every correct answer equally — the student gets an A for correctly identifying 1000 common coins without ever learning the rare ones. Focal loss is a teacher who says: getting the common coin right earns you 1 point, but getting the rare coin right earns you 6.4 points. Suddenly, the student has a reason to study the hard cases.*

*The γ parameter is how much the teacher penalises the student for practicing on coins they already know. A high γ says 'don't waste time on obvious coins.' Too high, and the student forgets what common coins look like entirely — they start flagging everything as rare (False Positives)."*

---

← [Architecture](02_architecture.md) | [Index](index.md) | Next: [Experiments →](04_experiments.md)
