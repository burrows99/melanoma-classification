# V. Conclusion and Future Work

## Conclusion

A dual-branch melanoma classifier fusing dermoscopic images with patient
metadata achieves F1=0.8984, recall=93.14% (Config B: Adam+CosineAnnealing),
reducing missed melanomas by 12.5% over the fixed-LR baseline with no
additional false positives. With TTA: F1=0.9134, recall=91.48%,
specificity=98.62%. A controlled ablation confirms metadata adds +1.9% recall;
SHAP identifies `anatom_site_torso` and `age_approx` as dominant predictors.
EigenCAM validates ABCDE-aligned attention. A Mahalanobis OOD detector provides
runtime safeguards.

## Future Work

1. **Larger backbones** (EfficientNet-B3/B5) [8] for increased capacity.
2. **Equity assessment** via Fitzpatrick phototype stratification [15].
3. **External validation** on the ISIC private test set.
4. **ViT/Swin backbones** [21, 22] with larger pretraining budgets.

---

# References

[1] R. L. Siegel et al., "Cancer statistics, 2020," *CA*, vol. 70(1), 2020.

[2] A. Esteva et al., "Dermatologist-level classification of skin cancer,"
*Nature*, vol. 542, pp. 115–118, 2017.

[3] H. Kittler et al., "Diagnostic accuracy of dermoscopy," *Lancet Oncol.*,
vol. 3(3), pp. 159–165, 2002.

[5] A. Esteva et al., "Dermatologist-level classification of skin cancer,"
*Nature*, vol. 542, pp. 115–118, 2017.

[6] N. Codella et al., "Skin lesion analysis toward melanoma detection 2018,"
arXiv:1902.03368, 2019.

[7] V. Rotemberg et al., "A patient-centric dataset of images and metadata for
identifying melanomas," *Sci. Data*, vol. 8(34), 2021.

[8] Q. Ha et al., "Identifying melanoma images using EfficientNet ensemble,"
arXiv:2010.05351, 2020.

[9] M. Tan and Q. V. Le, "EfficientNet: Rethinking model scaling," *ICML*,
2019, pp. 6105–6114.

[10] G. Huang et al., "Densely connected convolutional networks," *CVPR*,
2017, pp. 4700–4708.

[11] P. Rajpurkar et al., "CheXNet: Radiologist-level pneumonia detection,"
arXiv:1711.05225, 2017.

[12] K. He et al., "Deep residual learning for image recognition," *CVPR*,
2016, pp. 770–778.

[13] T.-Y. Lin et al., "Focal loss for dense object detection," *ICCV*, 2017,
pp. 2980–2988.

[14] B. C. L. Pacheco et al., "PAD-UFES-20: A skin lesion dataset," *Data in
Brief*, vol. 32, 2020.

[15] M. Groh et al., "Evaluating deep neural networks trained on clinical
images in dermatology with Fitzpatrick 17k," *CVPR Workshop*, 2021.

[16] Y. Zhang et al., "Self-supervised learning for medical image analysis,"
*Med. Image Anal.*, vol. 58, 2019.

[17] G. Wang et al., "Interactive medical image segmentation using deep
learning," *IEEE TMI*, vol. 37(7), pp. 1562–1573, 2018.

[18] Y. Fujisawa et al., "Deep-learning-based classifier surpasses
dermatologists," *Br. J. Dermatol.*, vol. 180(2), 2019.

[19] G. Brancaccio et al., "Dermoscopy and confocal microscopy patterns,"
*JEADV*, vol. 34, 2020.

[20] K. Lee et al., "A simple unified framework for detecting OOD samples,"
*NeurIPS*, 2018.

[21] A. Dosovitskiy et al., "An image is worth 16x16 words," *ICLR*, 2021.

[22] Z. Liu et al., "Swin Transformer," *ICCV*, 2021, pp. 10012–10022.

---

[← IV. Experiments](05_experiments.md)
