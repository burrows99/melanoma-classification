# Out-of-Distribution Detection (Mahalanobis)

← [Metadata & SHAP](07_metadata_shap.md) | [Index](index.md) | Next: [EigenCAM →](09_eigencam_interpretability.md)

---

## The Problem Without OOD Detection

A sigmoid output is unbounded confidence — the head *must* produce a number between 0 and 1 regardless of input. Feed it a photo of a koala:

1. EfficientNet-B0 fires on fur texture → activates filters similar to hair/pigmentation
2. The 1280-d embedding lands somewhere in malignant-class weight space
3. Sigmoid outputs **85.8% malignant confidence** — a confident, wrong answer

No error, no warning. The system would send a koala photo to a dermatologist with an 85.8% malignancy flag. This is a **deployment safety failure**.

---

## The Solution: Mahalanobis OOD Detector

**Core idea:** If an input's feature-space embedding is far from *any* valid training distribution embedding, it's probably out-of-distribution. Intercept it *before* trusting the logit.

### Step 1 — Build the reference distribution

At training time, run all **validation images** through the EfficientNet-B0 backbone (pre-fusion) to extract their **1280-d embeddings**.

Compute:
$$\mu = \frac{1}{N}\sum_{i=1}^N \mathbf{z}_i \quad \text{(mean embedding)}$$
$$\Sigma = \frac{1}{N}\sum_{i=1}^N (\mathbf{z}_i - \mu)(\mathbf{z}_i - \mu)^T \quad \text{(covariance matrix)}$$

### Step 2 — Compute Mahalanobis distance at inference

For a new input with embedding $\mathbf{z}$:

$$D_M(\mathbf{z}) = (\mathbf{z} - \mu)^T \Sigma^{-1} (\mathbf{z} - \mu)$$

> **Important:** The code computes the **squared** Mahalanobis distance (no square root) — `diff @ cov_inv @ diff` in `app.py`. The threshold is derived from the same squared distances (also via `dists.mean() + 3*dists.std()` in `evaluate.py`), so the comparison is internally consistent. The koala distance=5124 and threshold=2652 are therefore squared distances, not the textbook $\sqrt{\cdot}$ form. If asked in the viva: acknowledge this and note the system is functionally correct because both sides of the inequality use the same formula.

### Step 3 — Apply threshold

$$\text{threshold} = \mu_{D_M} + 3\sigma_{D_M}$$

If $D_M(\mathbf{z}) > \text{threshold}$ → **OOD warning issued**. The logit is not trusted.

---

## Grilling Question 1: "Why Mahalanobis instead of Euclidean distance?"

**Euclidean distance:**
$$D_E(\mathbf{z}) = \|\mathbf{z} - \mu\|_2 = \sqrt{\sum_{j=1}^{1280}(z_j - \mu_j)^2}$$

**Problems with Euclidean in 1280-d feature space:**
1. **Scale invariance:** Features may have very different variances. A feature that varies ±100 in training is treated as equally important as one that varies ±0.01.
2. **Independence assumption:** CNN features are highly correlated (nearby feature map channels fire together). Euclidean treats all 1280 dimensions as independent.

**Mahalanobis** corrects both via $\Sigma^{-1}$:
- The covariance matrix captures which dimensions vary together
- $\Sigma^{-1}$ *normalises* each dimension by its variance and *de-correlates* feature pairs
- Effectively, Mahalanobis measures distance in "standard deviation units" in the decorrelated feature space

**In English:** Euclidean is a ruler that treats all dimensions equally. Mahalanobis is a rubber ruler that stretches and squashes dimensions based on how much they normally vary in the training data.

---

## Grilling Question 2: "The koala example — give me the exact numbers."

- **Mahalanobis distance of koala:** 5124
- **Threshold for Exp 1:** 2652 (µ + 3σ of validation embedding distances)
- **Ratio:** 5124 / 2652 ≈ **1.93× the threshold** — nearly double
- **Raw sigmoid output (without OOD gate):** 85.8% malignant confidence

The detector intercepts at the **embedding level**, before the logit is produced or shown to the user.

**Threshold range across experiments:**
- Exp 3 (γ=1.5): 2637 — tightest threshold
- Baseline: 3248 — loosest threshold
- Cosine-annealed models produce tighter feature distributions (lower variance) → lower thresholds

---

## Grilling Question 3: "Why µ + 3σ as the threshold?"

Under a Gaussian assumption, µ ± 3σ captures **99.7%** of the training distribution. Anything beyond 3σ is in the tail — extremely unlikely to be a valid dermoscopic image. This is the empirical rule applied to the distribution of Mahalanobis distances across the validation set.

**Why not tighter (e.g., 2σ)?** You'd reject ~5% of valid validation images — weird lighting, unusual lesion morphologies that the CNN finds confusing but are legitimate dermoscopy.

**Why not looser (e.g., 4σ)?** More OOD inputs would slip through. The gap between a koala (5124) and the threshold (2652) is large enough that 3σ reliably intercepts non-dermoscopic images without being too aggressive.

**It's a tunable hyperparameter.** In production deployment, you'd calibrate this on a held-out OOD set.

---

## Grilling Question 4: "Where exactly does the detector intercept the pipeline?"

```
Image Input
     ↓
EfficientNet-B0 backbone
     ↓
1280-d embedding  ←── Mahalanobis distance computed HERE
     ↓                    ↓
  D_M > threshold?   → YES → "OOD warning: not a dermoscopic image"
     ↓ NO
Concatenate with metadata embedding (64-d)
     ↓
Linear head (1344→1)
     ↓
Sigmoid → probability
     ↓
Displayed to user
```

The check happens **before** the metadata branch and **before** the linear head. The OOD detector doesn't care about metadata — it only validates the visual embedding.

---

## Grilling Question 5: "What are the failure modes of this detector?"

**False OOD (valid image rejected):**
- Very unusual but real dermoscopic images (amelanotic melanoma, unusual body site)
- Close-up photos with non-standard framing
- Images from different camera types or lighting

**OOD images that slip through:**
- Images that happen to have texture/colour statistics similar to skin lesions (e.g., brown fur, wood grain)
- Images from a *different* medical imaging domain (e.g., histopathology slide — might land close to dermoscopy embeddings)

**The honest limitation:** The Mahalanobis detector is only as good as the validation set distribution. If validation doesn't cover the full range of valid dermoscopy variability, it sets the threshold too tightly. Future work: calibrate on a dedicated OOD test set.

---

## Layman Analogy

*"The Mahalanobis detector is a bouncer at a medical conference who knows what dermatologists and medical imaging equipment look like. The sigmoid head inside is a doctor who will diagnose anything that walks through the door — including a koala.*

*Euclidean distance is a bouncer who measures only height: anyone over 6'5" gets turned away. Mahalanobis distance is a bouncer who uses a full profile — height, weight, clothing style, ID badge — and knows that 'tall + lab coat + lanyard' is fine, but 'short + fur + no shoes' is suspicious. The covariance matrix is the bouncer's knowledge of how all these attributes relate to each other."*

---

← [Metadata & SHAP](07_metadata_shap.md) | [Index](index.md) | Next: [EigenCAM →](09_eigencam_interpretability.md)
