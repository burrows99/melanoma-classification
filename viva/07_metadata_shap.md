# Metadata Fusion & SHAP Analysis

← [Evaluation](06_evaluation_metrics.md) | [Index](index.md) | Next: [OOD Detection →](08_ood_detection.md)

---

## The Metadata Features

| Feature | Type | Preprocessing | Dimensions |
|---------|------|--------------|------------|
| `age_approx` | Continuous | Median imputation + z-score | 1 |
| `sex` | Categorical (M/F/Unknown) | Mode imputation + one-hot | 3 |
| `anatom_site_general_challenge` | Categorical (6 categories) | Mode imputation + one-hot | 10 |
| **Total** | | | **14-d vector** |

**MLP:** 14 → 128 → 64 (BatchNorm + ReLU + Dropout 0.3 after each layer).

---

## Grilling Question 1: "Prove the metadata isn't just adding noise."

**Three-pronged proof:**

**1. Ablation experiment (Exp 4):** Disabling the metadata branch while holding all other HPs constant causes FN to rise from 80 → 99 — 19 more missed cancers. This is a controlled experiment; the *only* change is removing the metadata MLP.

**2. SHAP analysis (KernelExplainer on Exp 1 validation set):** Computes the marginal contribution of each metadata feature to the final logit. If metadata were noise, SHAP values would be near zero. Instead:
- `anatom_site_torso` — highest mean |SHAP| value of all features
- `age_approx` — monotonic positive relationship (older → higher predicted risk)
- `anatom_site_anterior_torso` — rare but high-impact (SHAP up to 0.18 for individual samples)
- `sex` — modest but non-zero contribution

**3. Clinical coherence:** These SHAP-dominant features match known epidemiological risk factors for melanoma — age and anatomical site are standard clinical predictors. A model that learns "old patient + torso lesion → higher risk" is doing something medically meaningful, not memorising noise.

---

## Grilling Question 2: "What is SHAP and how does KernelExplainer work?"

**SHAP (SHapley Additive exPlanations)** — Lundberg & Lee (2017) — decomposes a model's prediction into per-feature contributions based on Shapley values from cooperative game theory.

**Shapley value for feature i:**
$$\phi_i = \sum_{S \subseteq F \setminus \{i\}} \frac{|S|!(|F| - |S| - 1)!}{|F|!} \left[f(S \cup \{i\}) - f(S)\right]$$

where F is the full feature set and $f(S)$ is the model's prediction using only subset S. In English: how much does feature i contribute to the prediction, averaged over all possible orderings of features?

**KernelExplainer** is model-agnostic — it doesn't need access to gradients. It:
1. Samples subsets of features (masking others with background values)
2. Runs each subset through the full model (CNN + MLP + fusion head)
3. Fits a locally linear explanation to the resulting predictions
4. Derives SHAP values from the linear model's coefficients

This is important because you can't use gradient-based SHAP (GradientExplainer) easily on the full fusion model — KernelExplainer treats it as a black box.

---

## Grilling Question 3: "Why is anatom_site_torso the dominant feature?"

**Epidemiological reason:** Torso lesions have high UV exposure (sunburn history) and are often large, flat, irregularly shaped lesions that can be missed during self-examination. The SIIM-ISIC dataset reflects real-world clinical prevalence where torso sites are disproportionately represented in confirmed melanoma cases.

**SHAP interpretation:** When `anatom_site_torso=1` (one-hot active), the model's logit is shifted significantly upward — the MLP has learned this is a strong prior risk signal. When it's 0 (not a torso site), the shift is downward.

**Drill-Down:** *Why is `anatom_site_anterior_torso` rare but high-impact?*  
Anterior torso (chest/abdomen front) is a specific sub-site with fewer samples. When it appears, the MLP assigns a very large SHAP value (up to 0.18) because it's learned this rare combination is strongly associated with malignancy in the training data. High impact + low frequency = the model has over-indexed on a small signal. This is a potential bias to acknowledge in limitations.

---

## Grilling Question 4: "The metadata added 39 FP. Is that acceptable?"

**It depends on clinical context:**

- **Screening tool:** You want high recall (catch everything). The +1.9pp recall (19 fewer FN) saving 19 patients outweighs 39 extra biopsy referrals. Biopsies are uncomfortable but safe.

- **Triage tool:** If every flagged case goes straight to surgery (not biopsy), the calculus changes. 39 extra surgeries is significant.

**For this system (screening/demo):** The recall gain is the right priority. A patient with a positive flag still goes to a dermatologist — the model doesn't diagnose, it *flags for review*.

**Quantification:** At 13.6% prevalence, +39 FP raises the false referral rate from 1.4% to 2.8% of benign cases. This is clinically manageable.

---

## Layman Analogy

*"SHAP is like asking a jury to vote on a verdict, but instead of a majority vote, you ask: 'how much did each piece of evidence change the outcome?' You add each clue to the table in every possible order and measure the average impact each clue has.*

*The metadata MLP is like the patient's GP file that the hospital has before the radiologist sees the scan. The radiologist doesn't see it during their analysis — but the consultant who makes the final call gets to read both the scan report and the GP file. The GP file (metadata) adds context that the scan alone can't capture."*

---

← [Evaluation](06_evaluation_metrics.md) | [Index](index.md) | Next: [OOD Detection →](08_ood_detection.md)
