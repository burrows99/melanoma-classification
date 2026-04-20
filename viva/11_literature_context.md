# Literature & Design Choices

← [Error Analysis](10_error_analysis.md) | [Index](index.md) | [Back to basics →](basic.md)

---

## The Key Papers (Know These Cold)

| Ref | Authors | Contribution | Why You Cite It |
|-----|---------|-------------|-----------------|
| [4] | Esteva et al., *Nature* 2017 | CNN matched 21 dermatologists | Foundational justification for DL in dermoscopy |
| [6] | Rotemberg et al., *Sci. Data* 2021 | ISIC 2020 dataset release | Your dataset — know it inside out |
| [7] | Ha et al., arXiv 2020 | EfficientNet ensemble won ISIC 2020 (AUC=0.949) | Justifies EfficientNet choice |
| [8] | Tan & Le, *ICML* 2019 | EfficientNet: compound scaling | Architecture backbone |
| [11] | Lin et al., *ICCV* 2017 | Focal loss for dense object detection | Your loss function |
| [15] | Fujisawa et al., *Br. J. Dermatol.* 2019 | Amelanotic/small lesions as DL failure modes | Validates your error analysis |
| [16] | Davis & Goadrich, *ICML* 2006 | AUC overstates utility on imbalanced data | Why you don't rely on AUC alone |
| [19] | Lee et al., *NeurIPS* 2018 | Mahalanobis OOD framework | Your OOD detector |

---

## Grilling Question 1: "Why not Vision Transformers (ViT/Swin)?"

**The argument against ViT at this scale:**

1. **Data requirement:** ViT has no *inductive biases* — it learns spatial relationships from scratch. CNNs assume translation equivariance (a feature looks the same regardless of position), which is a true assumption for dermoscopy lesions. ViT requires 10–100× more data to match CNN performance on medical imaging.

2. **Dataset size:** 37,648 training images is relatively small for ViT. The original ViT (Dosovitskiy et al., ICLR 2021) was pretrained on JFT-300M (300 million images). Even with ImageNet pretraining, ViT underperforms EfficientNet at this dataset scale.

3. **Practical:** The ISIC 2020 winning solution used EfficientNet — following the established domain winner is pragmatically justified.

**The honest counterargument (if asked):** ViT with large-scale medical pretraining (e.g., BioViL, Med-SAM) could outperform EfficientNet. This is explicitly listed as future work. The choice was a practical constraint, not a claim that CNNs are universally superior.

---

## Grilling Question 2: "Why not ResNet-50 or DenseNet-121?"

| Model | Params | ImageNet Acc | Argument Against |
|-------|--------|-------------|-----------------|
| ResNet-50 | 25.6M | 76.1% | 4.8× more params than EfficientNet-B0, lower accuracy — strictly worse tradeoff |
| DenseNet-121 | 8.0M | 74.4% | Dense connections → high memory usage; lower accuracy than EfficientNet |
| **EfficientNet-B0** | **5.3M** | **77.1%** | Best accuracy/efficiency; compound scaling; ISIC 2020 winner used this family |

**Compound scaling advantage:** When you want to scale up later (B1, B3, B5), you do so in a principled ratio-preserving way. ResNet depth-only scaling is a single-axis heuristic.

---

## Grilling Question 3: "Why the SIIM-ISIC 2020 dataset specifically?"

1. **Size:** 37k+ dermoscopic images — large enough for fine-tuning without GAN augmentation
2. **Metadata:** One of few datasets that includes *structured patient metadata* alongside images — essential for your dual-branch approach
3. **Labels:** Binary (malignant/benign), clinically validated, with biopsy confirmation
4. **Community benchmark:** ISIC challenge allows comparison with other published results (winning AUC=0.949 gives context for your AUC=0.99)
5. **Concatenation:** You added ISIC 2019 images to increase minority class samples — standard practice in melanoma classification

**Known limitation:** Dataset is predominantly from lighter-skinned patients (Groh et al., 2021 — Fitzpatrick dataset). Performance on darker skin tones is unknown — a real clinical equity concern.

---

## Grilling Question 4: "Focal loss was designed for object detection. Is it appropriate here?"

Lin et al. (2017) designed focal loss for **RetinaNet** — one-stage object detection where the vast majority of anchors are easy background negatives. The core problem is identical to yours: overwhelming majority class, rare hard positives.

The mathematical mechanism is class-agnostic — it suppresses easy examples regardless of whether "easy majority" means "background anchors" or "benign skin lesions." The α term was even described in the original paper as an inverse-frequency weight, exactly as you've applied it.

**The only adaptation needed:** Sigmoid (binary) instead of softmax (multi-class) output, which you've done.

---

## Grilling Question 5: "How does Pacheco et al. (2012) relate to your metadata fusion?"

**Pacheco et al. (2020)** — PAD-UFES-20 — created a multimodal skin lesion dataset explicitly pairing clinical images with patient data (age, skin lesion location, etc.) and demonstrated that models incorporating clinical context outperform image-only models.

This is the direct motivation for your late-fusion approach. Your SHAP results independently confirm their finding: age and anatomical site are significant predictors. You've replicated the key insight on a different, larger dataset (ISIC 2020 vs PAD-UFES-20).

---

## Grilling Question 6: "What is out of scope and why?"

| Out of scope | Reason |
|-------------|--------|
| GAN augmentation | Would generate synthetic malignant images; risk of mode collapse; out-of-scope for single HP ablation study |
| Ensembles | Would prevent isolating the effect of individual HP changes — confounds the ablation |
| Multi-backbone comparison | Would require identical training for each backbone; separate study; mentioned as future work |
| Multi-class diagnosis | ISIC 2020 binary labels only; multi-class requires different dataset (ISIC 2018) |
| ISIC private test set | Not available for post-challenge submission; external validation is future work |
| ViT / Swin | Data scale insufficient; explicitly acknowledged as future work |

---

## Grilling Question 7: "You cite AUC=0.99. Isn't that suspiciously high?"

**Honest answer:** It's high, but not unreasonable.

1. All five experiments hit AUC=0.99, which means it's stable across configurations — it's a property of the feature representation, not a lucky run.
2. The ISIC 2020 winning ensemble achieved AUC=0.949 on the *private test set* (harder). Our AUC on the validation set (same distribution as training) would naturally be higher.
3. AUC=0.99 with F1=0.8984 is consistent — high AUC means the model ranks well, but the operating threshold (0.5) doesn't fully exploit this. You're intentionally not over-optimising the threshold to keep comparisons clean.
4. Davis & Goadrich (2006) is cited precisely because high AUC can overstate clinical utility on imbalanced data — you already acknowledge this limitation.

---

## Layman Analogy

*"Choosing EfficientNet over ResNet is like choosing a fuel-efficient sports car over a gas-guzzling truck for a city race. The truck is bigger and more powerful (more parameters), but the sports car is faster and cheaper to run (better accuracy per parameter). The EfficientNet 'family' is also scalable — the same racing team (compound scaling formula) can build a bigger car (B3, B5) when you need more power.*

*Not using ViT is like not using a Formula 1 car for a city street race — incredible in the right conditions (massive data, specialist hardware) but impractical and slower than the sports car in our constrained setting."*

---

← [Error Analysis](10_error_analysis.md) | [Index](index.md) | [Back to basics →](basic.md)
