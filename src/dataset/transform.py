import torch
import torchvision.transforms.v2 as v2
import numpy as np
from PIL import Image
import albumentations as A

from config import Config


class Transform:
    """Unified image transform pipeline for both train and validation.

    Pass ``train=True`` for augmented training transforms,
    ``train=False`` for deterministic validation transforms.
    """

    def __init__(self, train: bool = True):
        self._train = train
        img_size = Config.get_model_config()['image_size']
        aug = Config.get_augmentation_config()

        if train:
            # albumentations: only for transforms without a torchvision.v2 equivalent
            self._alb = A.Compose([
                A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
                A.OneOf([
                    A.GaussianBlur(blur_limit=(3, 7), p=0.5),
                    A.MedianBlur(blur_limit=5, p=0.5),  # no torchvision equivalent
                ], p=0.3),
                A.RandomRotate90(p=0.5),  # snaps to 0/90/180/270°; v2 has no equivalent
            ])
            self._tv = v2.Compose([
                v2.Resize((img_size, img_size)),
                v2.RandomHorizontalFlip(p=aug.get('horizontal_flip_prob', 0.5)),
                v2.RandomVerticalFlip(p=aug.get('vertical_flip_prob', 0.5)),
                v2.RandomAffine(
                    degrees=aug.get('rotation', 20),
                    translate=(0.0625, 0.0625),
                    scale=(0.85, 1.15),
                ),
                v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.25, hue=0.083),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.RandomErasing(p=0.5, scale=(0.004, 0.016), ratio=(0.3, 3.3), value=0),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            self._alb = None
            self._tv = v2.Compose([
                v2.Resize((img_size, img_size)),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

    def __call__(self, pil_img):
        if self._train:
            assert self._alb is not None
            pil_img = Image.fromarray(self._alb(image=np.array(pil_img))['image'])
        return self._tv(pil_img)

    # TTA transforms applied to PIL images before the standard pipeline.
    # Image.Transpose enum avoids the deprecated integer constants.
    tta_transforms = {
        'original': lambda img: img,
        'hflip':   lambda img: img.transpose(Image.Transpose.FLIP_LEFT_RIGHT),
        'vflip':   lambda img: img.transpose(Image.Transpose.FLIP_TOP_BOTTOM),
        'rot90':   lambda img: img.rotate(90,  expand=False),
        'rot180':  lambda img: img.rotate(180, expand=False),
        'rot270':  lambda img: img.rotate(270, expand=False),
    }

