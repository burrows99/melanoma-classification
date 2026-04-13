import torch
import torch.nn as nn
import torchvision.models as tv_models
from config import MODEL_ARCHITECTURE, NUM_CLASSES

# Registry of supported architectures: name -> (model_fn, default_weights)
_BACKBONE_REGISTRY = {
    'efficientnet_b0': (tv_models.efficientnet_b0, tv_models.EfficientNet_B0_Weights.DEFAULT),
    'densenet121':     (tv_models.densenet121,     tv_models.DenseNet121_Weights.DEFAULT),
    'resnet50':        (tv_models.resnet50,         tv_models.ResNet50_Weights.DEFAULT),
}

def _build_backbone(arch: str, pretrained: bool):
    """Loads a torchvision backbone and strips its classifier head."""
    if arch not in _BACKBONE_REGISTRY:
        raise ValueError(f"Unsupported architecture '{arch}'. Supported: {list(_BACKBONE_REGISTRY)}")
    fn, default_weights = _BACKBONE_REGISTRY[arch]
    model = fn(weights=default_weights if pretrained else None)
    # Read feature size then replace head with Identity so forward() returns pooled features
    if hasattr(model, 'fc'):                          # ResNet
        num_features = model.fc.in_features
        model.fc = nn.Identity()
    elif hasattr(model, 'classifier'):                # EfficientNet / DenseNet
        head = model.classifier
        num_features = (head[-1] if isinstance(head, nn.Sequential) else head).in_features
        model.classifier = nn.Identity()
    else:
        raise AttributeError(f"Cannot locate classifier head for '{arch}'")
    return model, num_features


# --- Metadata Fusion Model ---
class MetadataMelanomaModel(nn.Module):
    """Model combining image features (CNN backbone) and tabular metadata (MLP)."""
    def __init__(self, num_metadata_features, pretrained=True,
                 cnn_dropout=0.0, mlp_hidden_dims=[128, 64], mlp_dropout=0.3):
        super().__init__()

        # Image Branch — pure torchvision backbone, classifier head removed
        self.image_backbone, self.num_image_features = _build_backbone(MODEL_ARCHITECTURE, pretrained)
        self.cnn_dropout = nn.Dropout(cnn_dropout) if cnn_dropout > 0 else nn.Identity()

        # Metadata Branch MLP
        layers = []
        input_dim = num_metadata_features
        for i, hidden_dim in enumerate(mlp_hidden_dims):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            # Apply dropout to all but last hidden layer output
            if i < len(mlp_hidden_dims) - 1: 
                layers.append(nn.Dropout(mlp_dropout))
            input_dim = hidden_dim # Output dim becomes input for next layer
        self.metadata_mlp = nn.Sequential(*layers)
        self.num_mlp_output_features = input_dim 

        # Final Classifier (on concatenated features)
        classifier_input_features = self.num_image_features + self.num_mlp_output_features
        self.final_classifier = nn.Linear(classifier_input_features, NUM_CLASSES)
        
    def forward(self, image_input, metadata_input):
        # Process image
        image_features = self.image_backbone(image_input)
        image_features = self.cnn_dropout(image_features)
        
        # Process metadata
        metadata_features = self.metadata_mlp(metadata_input)
        
        # Concatenate features
        combined_features = torch.cat((image_features, metadata_features), dim=1)
        
        # Final classification
        output = self.final_classifier(combined_features)
        return output

# --- Model Instantiation Function ---
def get_model(num_metadata_features):
    """Instantiates the metadata fusion model.
    Args:
        num_metadata_features (int): Number of features in the metadata input.
    Returns:
        torch.nn.Module: The instantiated model, moved to DEVICE.
    """
    print(f"Creating metadata fusion model '{MODEL_ARCHITECTURE}' with {num_metadata_features} metadata features.")
    model = MetadataMelanomaModel(num_metadata_features=num_metadata_features)
    model = model.to(DEVICE)
    return model