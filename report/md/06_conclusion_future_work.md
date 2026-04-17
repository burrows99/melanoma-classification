# V. Conclusion and Future Work

## Conclusion

This paper presented a dual-branch melanoma classification system fusing
dermoscopic images with patient metadata under severe class imbalance. Three
CNN backbones were compared under identical conditions. EfficientNet-B0 achieves
the best result: F1=0.9134, recall=91.48%, specificity=98.62% (TTA-enabled),
outperforming DenseNet-121 (F1=0.8969) and ResNet-50 (F1=0.8669). Its fastest
inference time (12.3 ms) makes it the most practical choice for clinical
screening. Metadata fusion contributes ~2.8% recall improvement over an
image-only baseline, with `age_approx` providing the strongest prior followed
by anatomical site and sex [7, 14, 16]. Focal loss with principled α=0.864
keeps all three backbones above 90% recall despite a 6.4:1 class imbalance.
Error analysis confirms that dominant failure modes—amelanotic melanomas (31%
of FN), small lesions <6 mm (28%)—align with known clinical challenges [18].
EigenCAM confirms attention aligns with ABCDE criteria, validating clinical
interpretability. The ISIC 2020 ensemble AUC of 0.9490 [8] is not reached as
expected: that system used 18 EfficientNet variants with TTA and external data
versus a single-model 20-epoch budget here.

## Future Work

1. **OOD detection.** A sigmoid classifier provides no domain guard; non-
   dermoscopic inputs can produce high malignancy probabilities. An explicit
   image-type detector is a necessary precondition for clinical deployment.
2. **Metadata feature importance.** SHAP or permutation analysis should
   formally quantify each field's contribution (age, sex, site) to the 2.8%
   recall gain.
3. **LR scheduling + larger backbones.** Cosine annealing would stabilise
   DenseNet-121 dynamics; scaling to EfficientNet-B3/B5 is the highest-return
   architectural change [8].
4. **Equity assessment.** Per Groh et al. [15], performance should be stratified
   by Fitzpatrick phototype before any clinical deployment.

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

[← IV. Experiments](05_experiments.md)
