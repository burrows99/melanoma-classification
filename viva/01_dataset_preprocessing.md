# Dataset & Preprocessing

← [Index](index.md) | Next: [Architecture →](02_architecture.md)

---

## The Facts

- **Source:** SIIM-ISIC 2020 + ISIC 2019 images concatenated → **37,648 total samples**
- **Image size:** 256×256 pixels
- **Class ratio:** **13.6% malignant**, 86.4% benign — ~6.4 benign per malignant
- **Split:** 80/20 stratified (seed 42), preserving the 13.6% ratio in both train and val
- **Validation set size:** ~7,530 samples

---

## Grilling Question 1: "Why a stratified split?"

**Answer:** With only 13.6% positives, a random split risks concentrating malignant samples in one partition. Stratified splitting guarantees both train and val see the same 13.6%:86.4% ratio, so validation metrics aren't artificially inflated or deflated by a lucky draw. Seed 42 makes it reproducible; all five experiments share the *identical* split so HP comparisons are fair.

**Drill-Down:** *What if you hadn't stratified?*  
A random split on 37k samples is actually fairly stable, but with a minority class you could easily get a val set with 11% or 16% positives — that shifts your recall/specificity numbers without the model actually changing. It would corrupt your ablation comparisons.

---

## Grilling Question 2: "Walk me through your augmentation pipeline exactly."

**Training images (two-stage):**

1. **Albumentations** (applied first, domain-specific):
   - **CLAHE** (Contrast Limited Adaptive Histogram Equalisation) — enhances local contrast in dermoscopic images, making lesion borders more visible
   - **Blur** — simulates camera defocus, prevents over-fitting to sharp edges
   - **90° snaps** — rigid rotations that preserve lesion morphology (dermoscopy has no canonical orientation)

2. **torchvision v2** (applied second, standard):
   - Random horizontal flip (p=0.5)
   - Random vertical flip (p=0.5)
   - Affine ±15° rotation
   - Colour jitter
   - Random erasing
   - ImageNet normalisation (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

**Validation images:** Resize + ImageNet normalisation only. No augmentation — you want a stable, deterministic benchmark.

**Why two stages?** Albumentations gives fine-grained medical-imaging transforms (CLAHE, elastic distortions); torchvision v2 gives well-optimised general-purpose augmentations.

---

## Grilling Question 3: "How did you preprocess the metadata?"

**Metadata features:**  
- `age_approx` → **continuous integer** (patient's approximate age in years). NaNs filled with the training-set median, then z-score standardised so it sits on the same scale as the one-hot features. Older age is a known melanoma risk factor — SHAP confirms a monotonic positive relationship.  
- `sex` → **categorical: male / female / unknown** (one-hot encoded to 3 binary columns). Sex contributes modestly to the final logit per SHAP; `unknown` is retained as its own column rather than dropped, preserving the missingness signal.  
- `anatom_site_general_challenge` → **6-category site label** (torso, lower extremity, upper extremity, head/neck, palms/soles, oral/genital — one-hot encoded to 10 columns including unknown). This is the dominant SHAP feature: `anatom_site_torso` consistently has the highest mean |SHAP value| because torso lesions are disproportionately represented in confirmed melanoma cases in this dataset.

**Result:** 14-dimensional vector fed into the metadata MLP.

**Why median imputation for age?** The median is robust to outlier ages and doesn't shift the distribution the way mean imputation does.  
**Why `'Unknown'` fill for categoricals?** The code explicitly fills NaN categoricals with the string `'Unknown'` (a sentinel/null category) rather than the statistical mode. This is intentional: it keeps missingness as a discrete, learnable category that the one-hot encoder handles cleanly, rather than silently injecting the most common class into missing rows which could bias SHAP values toward the dominant site.

---

## Grilling Question 4: "Why did you use CLAHE specifically?"

CLAHE is the standard preprocessing step in dermoscopy because it:
1. Enhances lesion borders relative to surrounding skin (where illumination gradients exist)
2. Uses a *clip limit* to prevent noise amplification in near-uniform regions
3. Works tile-by-tile (Adaptive HE) rather than globally, so it doesn't wash out local texture

Without it, the CNN first has to "waste" capacity learning to correct illumination variance before it can learn lesion features.

---

## Layman Analogy

*"Think of the 37k images as a box of 100 marbles where 86 are white and 14 are red. Stratified splitting is making sure that both 'practice' and 'test' piles have the same 14% red ratio — rather than randomly ending up with a test pile that's mostly red or mostly white. CLAHE is like turning on a good lamp before sorting: it makes it easier to see the difference between the marbles."*

---

← [Index](index.md) | Next: [Architecture →](02_architecture.md)
