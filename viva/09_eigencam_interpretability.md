# EigenCAM & Interpretability

← [OOD Detection](08_ood_detection.md) | [Index](index.md) | Next: [Error Analysis →](10_error_analysis.md)

---

## What EigenCAM Is

**EigenCAM** is a gradient-free saliency map method that computes the principal eigenvector of the feature map activations at a target convolutional layer. Unlike Grad-CAM (which needs gradients), EigenCAM:

1. Takes the feature map tensor $\mathbf{A} \in \mathbb{R}^{C \times H \times W}$ from the final conv layer
2. Reshapes to $\mathbf{A}' \in \mathbb{R}^{C \times HW}$
3. Computes SVD: $\mathbf{A}' = U \Sigma V^T$
4. Takes the first right singular vector $v_1$ — the direction of maximum variance in feature space
5. Reshapes $v_1$ back to $H \times W$ → **saliency map**

The result: a heatmap showing *where* in the image the network has concentrated the most informative activations.

---

## Grilling Question 1: "What does your EigenCAM show, and why does it matter?"

For the Exp 1 malignant lesion (Fig. 7 in the paper): the heatmap concentrates on:
- **Asymmetric borders** — one side of the lesion boundary is irregular
- **Colour variegation zones** — areas with mixed brown, black, and red pigmentation
- **Atypical pigment network** — disrupted skin surface texture within the lesion

These correspond precisely to the clinical **ABCDE criteria** for melanoma:
- **A**symmetry
- **B**order irregularity
- **C**olour variation
- **D**iameter (>6mm)
- **E**volution

**Why this matters:** It provides *post-hoc clinical validation*. A network that achieves 93.14% recall could theoretically be pattern-matching on irrelevant features (image artefacts, hair, dark corners). EigenCAM confirms the model is attending to the same features a dermatologist would — giving clinical confidence in the predictions.

---

## Grilling Question 2: "Why EigenCAM instead of Grad-CAM?"

**Grad-CAM** weights feature map channels by the gradient of the target class score with respect to those channels:
$$\alpha_k = \frac{1}{Z}\sum_{i,j}\frac{\partial y^c}{\partial A_{ij}^k} \quad \text{(global average pooled gradient)}$$
$$L^c = \text{ReLU}\left(\sum_k \alpha_k A^k\right)$$

**Problems with Grad-CAM in this setting:**
1. Requires a differentiable path from output to target layer — complex with the dual-branch fusion and focal loss
2. Gradient-based methods can be noisy for small lesions
3. Sensitive to gradient saturation in ReLU networks

**EigenCAM advantages:**
- No gradients needed — works on any CNN layer
- More stable: eigenvectors are smoother than raw gradients
- Faster to compute at inference (useful for the Gradio demo)

**Limitation:** EigenCAM is not class-discriminative — it shows where the network is most active, not specifically which features drove the malignant vs benign decision. For class discrimination, Grad-CAM++ or ScoreCAM would be more precise. This is an honest limitation to acknowledge.

---

## Grilling Question 3: "Which layer did you apply EigenCAM to, and why?"

Applied to the **final convolutional layer** of EfficientNet-B0 (before global average pooling). This is standard practice for saliency methods because:
- **Early layers:** Learn low-level features (edges, colours) — saliency maps too diffuse
- **Middle layers:** Learn mid-level textures — not directly interpretable
- **Final conv layer:** Highest-level semantic features before spatial information is collapsed by global average pooling — best spatial resolution while retaining semantic meaning

After global average pooling, spatial information is gone — no saliency map is possible.

---

## Grilling Question 4: "Could the model be attending to the right region for the wrong reason?"

Yes — this is called **Clever Hans behaviour** (Lapuschkin et al., 2019). A model could attend to the dark vignette common in dermoscopy images, or to ruler marks, or to hair — and still achieve high accuracy because these artefacts correlate with the presence of unusual lesions.

**What you did to mitigate this:**
1. **CLAHE preprocessing** removes global illumination gradients
2. **Random erasing augmentation** forces the model to not rely on any single region
3. **EigenCAM qualitative check** — the visualisations show lesion-centred attention, not border/background attention

**Limitation:** You haven't done a systematic adversarial test (e.g., occluding the lesion and checking if the model changes its prediction). This is a fair criticism and future work.

---

## Grilling Question 5: "What do the saliency maps look like for false negatives?"

For the types of FN cases (amelanotic melanoma, small lesions):
- Amelanotic melanoma: the EigenCAM heatmap is *diffuse* — the network can't find a concentrated region of discriminative colour features because the lesion lacks pigmentation
- Small lesions (<6mm): the heatmap may spread over surrounding skin texture rather than focusing on the tiny lesion

This visually confirms the error analysis — it's not a random failure, it's a systematic weakness tied to the absence of the visual cues the CNN relies on.

---

## Layman Analogy

*"EigenCAM is like asking the model 'show me what you were looking at when you made this decision.' It highlights the regions of the image that triggered the strongest activations in the final layer of the CNN.*

*Think of it as thermal goggles for the neural network's attention. Hot spots (red) = 'I was really focusing here.' Cold spots (blue) = 'I wasn't paying attention here.' If the hot spots are on the lesion's irregular border rather than on the patient's hair or the background, we know the model is looking at the right thing — the same things a dermatologist would examine."*

---

← [OOD Detection](08_ood_detection.md) | [Index](index.md) | Next: [Error Analysis →](10_error_analysis.md)
