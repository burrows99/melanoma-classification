# --- Configuration File ---

import torch
import torchvision.models as tv_models


class Config:
    _overrides: dict = {}

    @classmethod
    def override(cls, **kwargs) -> None:
        """Apply runtime overrides; called once from main.py before any config is read."""
        cls._overrides.update(kwargs)

    @classmethod
    def _merge(cls, base: dict) -> dict:
        """Return base dict with any matching overrides applied."""
        return {**base, **{k: cls._overrides[k] for k in base if k in cls._overrides}}

    MODEL_NAME    = 'efficientnet_b0'
    BACKBONE_FN   = staticmethod(tv_models.efficientnet_b0)
    BACKBONE_WEIGHTS = tv_models.EfficientNet_B0_Weights.DEFAULT

    # --experiment presets (from improvements.md)
    # Baseline (no flag): Adam, no scheduler, WD=0, γ=2.0
    EXPERIMENT_PRESETS = {
        1: {  # B — Cosine LR
            'optimizer': 'Adam',
            'weight_decay': 0,
            'scheduler': 'CosineAnnealingLR',
            'focal_gamma': 2.0,
        },
        2: {  # C — AdamW + cosine
            'optimizer': 'AdamW',
            'weight_decay': 1e-5,
            'scheduler': 'CosineAnnealingLR',
            'focal_gamma': 2.0,
        },
        3: {  # D — AdamW + cosine + γ=1.5
            'optimizer': 'AdamW',
            'weight_decay': 1e-5,
            'scheduler': 'CosineAnnealingLR',
            'focal_gamma': 1.5,
        },
        4: {  # Image-only ablation — no metadata branch
            'optimizer': 'Adam',
            'weight_decay': 0,
            'scheduler': None,
            'focal_gamma': 2.0,
            'image_only': True,
        },
    }

    _experiment: int | None = None

    @classmethod
    def set_experiment(cls, experiment: int) -> None:
        cls._experiment = experiment

    @classmethod
    def get_experiment(cls) -> int | None:
        return cls._experiment

    @classmethod
    def get_experiment_config(cls) -> dict | None:
        if cls._experiment is None:
            return None
        return cls.EXPERIMENT_PRESETS[cls._experiment]

    @classmethod
    def get_model_config(cls):
        return cls._merge({
            'image_size' : 256,
            'num_metadata_features' : 14,
        })

    @classmethod
    def get_training_config(cls):
        return cls._merge({
            'learning_rate' : 1e-4,
            'batch_size' : 32,
            'num_epochs' : 20,
            'train_split' : 0.8,
            'random_seed' : 42,
            'num_workers' : 4,
            'device' : 'cuda' if torch.cuda.is_available() else 'cpu',
        })

    @staticmethod
    def get_metadata_config():
        return {
            'use_metadata':     True,
            'metadata_cols':    ['sex', 'age_approx', 'anatom_site_general_challenge'],
            'numerical_cols':   ['age_approx'],
            'categorical_cols': ['sex', 'anatom_site_general_challenge'],
            'defaults': {
                'age_approx':                       50.0,
                'sex':                              'male',
                'anatom_site_general_challenge':    'torso',
            },
        }

    @classmethod
    def get_paths_config(cls):
        return cls._merge({
            'train_data_dir':    'dataset/train',
            'train_labels_path': 'dataset/train_concat.csv',
            'kaggle_dataset':    'nroman/melanoma-external-malignant-256',
            'hf_model_repo':     'burrows99/melanoma-models',
        })

    @classmethod
    def get_loss_config(cls):
        # Focal loss is used due to heavy class imbalance (~13.6% malignant).
        # Standard cross-entropy is dominated by the easy benign majority;
        # focal loss down-weights those so training focuses on hard/rare positives.
        # alpha = 1 - 0.136 (inverse malignant proportion) -> upweights positive examples.
        exp = cls.get_experiment_config()
        gamma = exp['focal_gamma'] if exp else 2.0
        return {
            'type':         'FocalLoss',
            'alpha':        0.864,
            'gamma':        gamma,
            'reduction':    'mean',
        }

    @staticmethod
    def get_augmentation_config():
        return {
            'rotation':             15,
            'horizontal_flip_prob': 0.5,
            'vertical_flip_prob':   0.5,
            'color_jitter': {
                'brightness': 0.2,
                'contrast':   0.2,
                'saturation': 0.1,
                'hue':        0.05,
            },
        }

    @classmethod
    def get_evaluation_config(cls):
        return cls._merge({
            'tta_enabled': False,  # Set to True to enable TTA during evaluation
        })




