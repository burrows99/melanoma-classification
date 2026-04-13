import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
from PIL import Image
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

import albumentations as A
from albumentations.pytorch import ToTensorV2

# Import necessary configurations only
from config import (
    IMAGE_SIZE, AUGMENTATION, TRAIN_LABELS_PATH, TRAIN_DATA_DIR, 
    TRAIN_SPLIT, RANDOM_SEED, BATCH_SIZE, NUM_WORKERS, DEVICE
)

# --- Metadata Configuration ---
METADATA_COLS = ['sex', 'age_approx', 'anatom_site_general_challenge']
NUMERICAL_COLS = ['age_approx']
CATEGORICAL_COLS = ['sex', 'anatom_site_general_challenge']


# --- Dataset Class ---
class MelanomaDataset(Dataset):
    """Dataset for melanoma images and associated metadata."""
    def __init__(self, image_paths, labels, metadata_features, transform=None):
        self.image_paths = image_paths
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.metadata_features = torch.tensor(metadata_features, dtype=torch.float32)
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            # Basic error handling for image loading issues.
            print(f"Warning: Error loading image {img_path}: {e}. Skipping item (returning first item)." )
            # Note: Fallback returns the first item; label/metadata may not match the returned image!
            idx = 0 
            img_path = self.image_paths[idx]
            image = Image.open(img_path).convert('RGB')
            
        label = self.labels[idx]
        metadata = self.metadata_features[idx]

        if self.transform:
            image = self.transform(image)
            
        return image, metadata, label

# --- Albumentations Transform Class (for pickling with multiprocessing) ---
class AlbumentationsTrainTransform:
    def __init__(self):
        # Normalize is a torchvision transform, applied after ToTensorV2
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                            std=[0.229, 0.224, 0.225])
        # Albumentations pipeline
        self.alb_transforms = A.Compose([
            A.Resize(IMAGE_SIZE, IMAGE_SIZE), # Ensure consistent size
            A.HorizontalFlip(p=AUGMENTATION.get('horizontal_flip_prob', 0.5)),
            A.VerticalFlip(p=AUGMENTATION.get('vertical_flip_prob', 0.5)),
            A.ShiftScaleRotate(
                shift_limit=0.0625, 
                scale_limit=0.15, 
                rotate_limit=AUGMENTATION.get('rotation', 20), 
                p=0.75,
                border_mode=0 
            ),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.75),
            A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=25, val_shift_limit=15, p=0.5),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 7), p=0.5),
                A.MedianBlur(blur_limit=5, p=0.5),
            ], p=0.3),
            A.CoarseDropout(
                max_holes=8, max_height=IMAGE_SIZE//8, max_width=IMAGE_SIZE//8, 
                min_holes=1, min_height=IMAGE_SIZE//16, min_width=IMAGE_SIZE//16, 
                fill_value=0, 
                p=0.5
            ),
            ToTensorV2()
        ])

    def __call__(self, pil_img):
        np_img = np.array(pil_img)
        augmented = self.alb_transforms(image=np_img) 
        img_tensor = augmented['image'] # Output from ToTensorV2 is CHW torch.uint8 tensor [0-255]
        
        # Convert tensor to float and scale to [0.0, 1.0] range before applying normalization
        img_tensor = img_tensor.float() / 255.0 
        
        return self.normalize(img_tensor)

# --- Transformations (Image) ---
def get_image_transforms(train=True):
    """Get image transformation pipeline."""
    if train:
        return AlbumentationsTrainTransform() # Uses the custom class with Albumentations
    else:
        # Validation/Test: Uses simpler torchvision transforms
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                         std=[0.229, 0.224, 0.225])
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(), # Converts PIL HWC [0-255] uint8 to CHW [0.0-1.0] float32 tensor
            normalize
        ])

# --- Metadata Preprocessor ---
def get_metadata_preprocessor(train_df):
    """Creates a sklearn preprocessor pipeline for metadata.
       Fits only on the training data provided.
    """
    numerical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    preprocessor = ColumnTransformer([
        ('num', numerical_pipeline, NUMERICAL_COLS),
        ('cat', categorical_pipeline, CATEGORICAL_COLS)
    ], remainder='drop')
    preprocessor.fit(train_df[METADATA_COLS])
    try:
        num_output_features = preprocessor.transform(train_df[METADATA_COLS][:1]).shape[1]
    except Exception as e:
        print(f"Could not determine metadata output features: {e}.")
        # Fallback value if determination fails
        num_output_features = 15 
        print(f"Warning: Using fallback metadata feature count: {num_output_features}")
    print(f"Metadata preprocessor created. Output features: {num_output_features}")
    return preprocessor, num_output_features

# --- Data Loaders ---
def get_data_loaders():
    """Load data, preprocess metadata, split, and create DataLoaders."""
    try:
        df = pd.read_csv(TRAIN_LABELS_PATH)
    except FileNotFoundError:
        print(f"Error: Labels CSV not found at {TRAIN_LABELS_PATH}")
        raise

    image_paths = [os.path.join(TRAIN_DATA_DIR, f"{img_name}.jpg") for img_name in df['image_name']]
    labels = df['target'].values
    indices = list(range(len(df)))

    train_indices, val_indices = train_test_split(
        indices, train_size=TRAIN_SPLIT, random_state=RANDOM_SEED, stratify=labels
    )

    train_df = df.iloc[train_indices]
    val_df = df.iloc[val_indices]

    # Preprocess metadata using the fitted preprocessor
    metadata_preprocessor, num_metadata_features = get_metadata_preprocessor(train_df)
    train_metadata_processed = metadata_preprocessor.transform(train_df[METADATA_COLS]).toarray()
    val_metadata_processed = metadata_preprocessor.transform(val_df[METADATA_COLS]).toarray()

    train_img_paths = [image_paths[i] for i in train_indices]
    val_img_paths = [image_paths[i] for i in val_indices]
    train_labels_final = train_df['target'].values
    val_labels_final = val_df['target'].values

    print(f"Dataset split: {len(train_img_paths)} train, {len(val_img_paths)} validation samples.")

    train_dataset = MelanomaDataset(
        train_img_paths, train_labels_final, train_metadata_processed,
        transform=get_image_transforms(train=True)
    )
    val_dataset = MelanomaDataset(
        val_img_paths, val_labels_final, val_metadata_processed,
        transform=get_image_transforms(train=False)
    )

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True
    )

    return train_loader, val_loader, num_metadata_features 