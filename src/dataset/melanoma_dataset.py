import torch
from typing import override
from torch.utils.data import Dataset
from PIL import Image


class MelanomaDataset(Dataset):
    """Dataset for melanoma images and associated metadata."""
    def __init__(self, image_paths, labels, metadata_features, transform=None):
        self.image_paths = image_paths
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.metadata_features = torch.tensor(metadata_features, dtype=torch.float32)
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    @override
    def __getitem__(self, idx: int) -> tuple: # ty:ignore[invalid-method-override]
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
