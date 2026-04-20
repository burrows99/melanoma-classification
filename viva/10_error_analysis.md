# Error Analysis — False Negatives & False Positives

← [EigenCAM](09_eigencam_interpretability.md) | [Index](index.md) | Next: [Literature →](11_literature_context.md)

---

## The Confusion Matrix for Exp 1 (Best Model)

|  | Predicted Benign | Predicted Malignant |
|--|-----------------|-------------------|
| **Actual Benign** | TN = ~6,315 | FP = **145** |
| **Actual Malignant** | FN = **70** | TP = ~1,000 |

From 7,530 validation samples (13.6% malignant ≈ 1,024 true malignant).

- **FN rate:** 70 / 1024 ≈ **6.8% of true malignant cases missed**
- **FP rate:** 145 / 6,506 ≈ **2.2% of true benign cases falsely flagged**

---

## False Negatives (The Dangerous Misses)

### What types of lesions are missed?

**1. Amelanotic melanoma (~3-4% of all melanoma)**
- No visible pigmentation — appears pink or skin-coloured
- The CNN relies on colour contrast and pigmentation patterns from training data
- Amelanotic lesions have no ABCDE-detectable colour feature
- EigenCAM shows diffuse, unfocused activation — the network has nothing to "grip"

**2. Small lesions (<6mm)**
- Below the threshold where ABCDE criteria become visible
- At 256×256 resolution, a 6mm lesion on a full-body shot may be <20 pixels across
- The CNN spatial resolution isn't fine enough to detect micro-features

**3. Early superficial spreading melanoma**
- Very flat, slightly irregular — early stage lesions have subtle borders
- Before colour variegation develops, these look like atypical benign naevi
- Identified by Fujisawa et al. (2019) as a systematic DL failure mode

### Grilling Question: "What would you do to reduce FN further?"

1. **Multi-scale input:** Run the image at multiple resolutions — catch small lesions at higher zoom
2. **Dedicated augmentation for amelanotic cases:** Simulate desaturation during training
3. **EfficientNet-B3/B5:** Larger models with higher native resolution → better small-lesion detection
4. **Ensemble:** Average multiple models — errors tend to be uncorrelated
5. **Threshold calibration:** Lower the decision threshold from 0.5 to 0.3 → catches more malignant at cost of more FP (acceptable for screening)

---

## False Positives (The False Alarms)

### What types of benign lesions are misclassified?

**1. Dysplastic (atypical) naevi**
- Benign moles with irregular borders, colour variation, asymmetry
- Share ABCDE features with melanoma — clinically ambiguous even for dermatologists
- The model and the dermatologist are making the same mistake — this is a feature, not a bug

**2. Atypical Spitz naevi**
- A specific benign variant that histologically resembles early melanoma
- Even dermatopathologists sometimes disagree on classification
- The metadata (age + site) may push the model toward malignant for young patients with torso lesions — appropriate caution

**3. Irregular seborrhoeic keratoses**
- Waxy, "stuck-on" benign lesions that can develop dark, irregular pigmentation
- The CNN's colour and texture detectors fire on the darkened, irregular regions

### Grilling Question: "Are FP always wrong to flag?"

**No — many FP are clinically defensible.** Dysplastic naevi and atypical Spitz naevi warrant biopsy in real clinical practice. The model is not making random errors; it's flagging the *hard cases* — the same cases that would trigger a dermatologist's concern. A 2.2% false alarm rate for cases that need biopsy anyway is clinically reasonable.

---

## Experiment-by-Experiment FN/FP Story

| Config | FN | FP | Clinical Story |
|--------|----|----|---------------|
| Baseline | 80 | 145 | Reasonable baseline — each error class manageable |
| Exp 1 | **70** | 145 | Best: 10 fewer missed cancers, same false alarms |
| Exp 2 | 66 | **181** | Best recall, but 36 extra unnecessary referrals |
| Exp 3 | 74 | **144** | Specificity rescue — fewest FP with metadata, at cost of 8 FN |
| Exp 4 | 99 | 106 | Image-only: fewest FP (no metadata sensitivity) but 19 more deaths |

### Which is the "best" experiment clinically?

- **Screening tool (catch everything):** Exp 2 — highest recall (93.54%), accept 181 FP
- **Balanced tool:** Exp 1 — best F1 (0.8984), saves 10 cancers vs Baseline, no FP penalty
- **Specificity-critical (minimise referrals):** Exp 3 — fewest FP with metadata (144), acceptable FN (74)
- **Never Exp 4** clinically — 19 extra dead patients is not a tradeoff for 39 fewer referrals

---

## Grilling Question: "Your FN rate is 6.8%. Is that acceptable?"

**Context:**

- Human dermatologist miss rate in dermoscopy: varies 10–20% without decision support (Kittler et al., Lancet Oncol 2002 — dermoscopy improves sensitivity by 10–27pp over naked-eye)
- AI-assisted systems in literature: typically 5–15% FN depending on dataset and subtype
- **6.86% FN rate is competitive for a single-model screen**

The system is positioned as a **screening tool** (flag for expert review), not a diagnostic decision-maker. At this operating point, 93.14% recall is a meaningful contribution — catching 93 in 100 cases, with a dermatologist catching the remaining 7.

---

## Layman Analogy

*"FN are the fish that slip through the net — the ones you were trying to catch. Amelanotic melanoma is an invisible fish: your net (CNN) is designed to catch fish with a certain colour, and this fish is clear. Small lesions are tiny fish that slip through the mesh.*

*FP are small rocks that look like fish and get caught unnecessarily — dysplastic naevi look enough like melanoma to fool the net. But these rocks (atypical naevi) are usually worth inspecting anyway; a dermatologist can release them back quickly.*

*The goal is: minimise invisible fish escaping (FN), tolerate some rocks in the catch (FP), because a missed invisible fish is fatal, but an extra rock in the net is just a minor inconvenience."*

---

← [EigenCAM](09_eigencam_interpretability.md) | [Index](index.md) | Next: [Literature →](11_literature_context.md)
