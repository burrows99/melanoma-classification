# V. Conclusion and Future Work

## Conclusion

This paper presented a multi-modal melanoma classification system that fuses
dermoscopic image features from a pretrained CNN backbone with structured patient
metadata—age, sex, and anatomical site—under a severely class-imbalanced setting.
Motivated by the patient-centric design of the ISIC 2020 challenge [7] and the
explicit demonstration by Rotemberg et al. that contextual clinical metadata
improves diagnostic discrimination, a dual-branch late-fusion architecture was
designed, implemented in PyTorch, and systematically evaluated across three
backbone choices.

EfficientNet-B0 is the best-performing architecture, achieving a validation F1
of **0.9134**, recall of **91.48%**, and specificity of **98.62%** with TTA
enabled (F1=0.9076 without TTA), outperforming DenseNet-121 (F1=0.8969) and
ResNet-50 (F1=0.8669) under the same hyperparameter search. EfficientNet-B0 also
offers the fastest inference time (12.3 ms vs. 17.8 ms for DenseNet-121 and
15.4 ms for ResNet-50), making it the most practical choice for real-time
clinical screening. The metadata branch contributes approximately **2.8%
improvement in recall** over an image-only baseline (Section IV, Table I), with
`age_approx` providing the strongest prior, followed by `anatom_site` and
`sex` [7, 14, 16]. Hyperparameter optimisation yields a further +0.025 F1 gain
for EfficientNet-B0; DenseNet-121 and ResNet-50 show no improvement, suggesting
they require a different regularisation strategy (e.g., cosine LR decay) beyond
the searched grid.

Sigmoid focal loss with α=0.864 (mathematically derived from the inverse
malignant class proportion) and γ=2.0 is effective: all three backbones maintain
validation recall above 90% despite a 6.4:1 class imbalance. Structured error
analysis confirms that the dominant false-negative categories—amelanotic
melanomas (31%), small lesions <6mm (28%), and early superficial spreading type
(23%)—align with the known limitations of dermoscopy-based AI identified by
Fujisawa et al. [18], validating that the model faces clinically realistic rather
than artificially simple failure modes. EigenCAM visualisations confirm that
model attention correctly localises to ABCDE-relevant lesion features rather than
background artefacts.

## Future Work

1. **OOD detection.** The sigmoid classifier provides no guarantee for
   out-of-domain inputs: a non-dermoscopic image can produce a high malignancy
   probability because the CNN maps all inputs into the same feature space. An
   explicit image-type detector or energy-based OOD score [Nguyen et al., 2015]
   is a necessary safeguard before any clinical deployment.

2. **Metadata feature importance.** Permutation importance or SHAP value
   analysis should formally quantify the individual contribution of each
   metadata field (age, sex, anatomical site) to the 2.8% recall improvement,
   informing which fields are critical to collect in a clinical workflow.

3. **Learning-rate scheduling.** Cosine annealing or ReduceLROnPlateau would
   dampen DenseNet-121's oscillatory dynamics and may reveal the full potential
   of the coarse hyperparameter grid search for ResNet-50.

4. **Self-supervised pretraining.** Zhang et al. [16] show that SSL pretraining
   on unlabelled dermoscopic images combined with metadata fusion yields
   consistent AUC gains; applying this to our backbone would complement the
   current supervised training regime.

5. **Equity and bias assessment.** Following Groh et al. [15], performance
   should be stratified by Fitzpatrick phototype before any clinical deployment,
   particularly given the dataset's known bias toward fair skin types.

6. **Larger EfficientNet variants.** Scaling to B3 or B5 within the same
   training budget—as done in the winning ISIC ensemble [8]—represents the
   highest-return single architectural change.

---

# References

[1] R. L. Siegel, K. D. Miller, and A. Jemal, "Cancer statistics, 2020," *CA:
A Cancer Journal for Clinicians*, vol. 70, no. 1, pp. 7–30, 2020.

[2] A. Esteva et al., "Dermatologist-level classification of skin cancer with
deep neural networks," *Nature*, vol. 542, pp. 115–118, 2017.

[3] H. Kittler et al., "Diagnostic accuracy of dermoscopy," *The Lancet
Oncology*, vol. 3, no. 3, pp. 159–165, 2002.

[4] A. C. Geller et al., "Melanoma epidemiology and prevention," *Clinical
Dermatology*, vol. 38, no. 5, pp. 591–597, 2020.

[5] A. Esteva et al., "Dermatologist-level classification of skin cancer with
deep neural networks," *Nature*, vol. 542, pp. 115–118, 2017.

[6] N. Codella et al., "Skin lesion analysis toward melanoma detection 2018:
A challenge hosted by the International Skin Imaging Collaboration (ISIC),"
arXiv:1902.03368, 2019.

[7] V. Rotemberg et al., "A patient-centric dataset of images and metadata for
identifying melanomas using clinical context," *Scientific Data*, vol. 8,
no. 34, 2021. arXiv:2008.07360.

[8] Q. Ha, B. Liu, and F. Liu, "Identifying melanoma images using EfficientNet
ensemble: Winning solution to the SIIM-ISIC melanoma classification challenge,"
arXiv:2010.05351, 2020.

[9] M. Tan and Q. V. Le, "EfficientNet: Rethinking model scaling for
convolutional neural networks," in *Proc. ICML*, 2019, pp. 6105–6114.

[10] G. Huang, Z. Liu, L. van der Maaten, and K. Q. Weinberger, "Densely
connected convolutional networks," in *Proc. CVPR*, 2017, pp. 4700–4708.

[11] P. Rajpurkar et al., "CheXNet: Radiologist-level pneumonia detection on
chest X-rays with deep learning," arXiv:1711.05225, 2017.

[12] K. He, X. Zhang, S. Ren, and J. Sun, "Deep residual learning for image
recognition," in *Proc. CVPR*, 2016, pp. 770–778.

[13] T.-Y. Lin, P. Goyal, R. Girshick, K. He, and P. Dollár, "Focal loss for
dense object detection," in *Proc. ICCV*, 2017, pp. 2980–2988.

[14] A. G. C. Pacheco et al., "PAD-UFES-20: A skin lesion dataset composed of
patient data and clinical images collected from smartphones," *Data in Brief*,
vol. 32, p. 106221, 2020.

[15] M. Groh et al., "Evaluating deep neural networks trained on clinical images
in dermatology with the Fitzpatrick 17k dataset," in *Proc. CVPR Workshops*,
2021, pp. 1820–1828. arXiv:2104.09957.

[16] R. Zhang, M. Guillemot, A. Mansoor, and O. Gevaert, "Enhancing melanoma
classification with metadata integration and self-supervised pretraining,"
arXiv:2312.04189, 2023.

[17] S. Minaee, M. Khosravi, and P. W. Wang, "Test-time augmentation for deep
learning-based cell segmentation on microscopy images," *Journal of Imaging*,
vol. 11, no. 1, p. 15, 2023. doi:10.3390/jimaging11010015.

[18] Y. Fujisawa, S. Inoue, and Y. Nakamura, "Deep learning for melanoma
diagnosis: The limitations and the necessity for a new strategy," *Frontiers in
Medicine*, vol. 8, p. 643302, 2021. doi:10.3389/fmed.2021.643302.

[19] G. Brancaccio, A. Balato, J. Malvehy, S. Puig, G. Argenziano, and
H. Kittler, "Artificial intelligence in skin cancer diagnosis: A reality check,"
*J. Invest. Dermatol.*, vol. 144, 2023. doi:10.1016/j.jid.2023.10.004.

---

| | |
|---|---|
| [← IV. Experiments](05_experiments.md) | |
