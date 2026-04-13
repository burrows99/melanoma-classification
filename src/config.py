# --- Configuration File ---

import torch


class Config:

    @staticmethod
    def get_model_config():
        return {
            'architecture' : 'efficientnet_b0',
            'num_classes' : 1,
            'image_size' : 256,
            'num_metadata_features' : 14,
        }

    @staticmethod
    def get_training_config():
        return {
            'learning_rate' : 1e-4,
            'batch_size' : 32,
            'num_epochs' : 20,
            'train_split' : 0.8,
            'random_seed' : 42,
            'num_workers' : 4,
            'device' : 'cuda' if torch.cuda.is_available() else 'cpu',
        }

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

    @staticmethod
    def get_paths_config():
        return {
            'train_data_dir':    r"E:\APML\Datasets\Melanoma_external_256\train",
            'train_labels_path': r"E:\APML\Datasets\Melanoma_external_256\train_concat.csv",
        }

    @staticmethod
    def get_loss_config():
        # Focal loss is used due to heavy class imbalance (~13.6% malignant).
        # Standard cross-entropy is dominated by the easy benign majority;
        # focal loss down-weights those so training focuses on hard/rare positives.
        # alpha = 1 - 0.136 (inverse malignant proportion) -> upweights positive examples.
        return {
            'type':         'FocalLoss',
            'alpha':        0.864,
            'gamma':        2.0,
            'reduction':    'mean',
            'class_weights': [1.0, 4.0],  # kept for reference, not used with Focal Loss
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

    @staticmethod
    def get_evaluation_config():
        return {
            'tta_enabled': False,  # Set to True to enable TTA during evaluation
        }




