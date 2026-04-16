import logging
import torch
import torch.nn as nn
import torchvision.models as tv_models
from torchvision.ops import MLP, sigmoid_focal_loss
from functools import partial
from config import Config

logger = logging.getLogger(__name__)


# --- Metadata Fusion Model ---
class MetadataMelanomaModel(nn.Module):
    """Model combining image features (CNN backbone) and tabular metadata (MLP)."""

    # Supported torchvision backbones: architecture name -> (factory fn, pretrained weights)
    _BACKBONES = {
        'efficientnet_b0': (tv_models.efficientnet_b0, tv_models.EfficientNet_B0_Weights.DEFAULT),
        'densenet121':     (tv_models.densenet121,     tv_models.DenseNet121_Weights.DEFAULT),
        'resnet50':        (tv_models.resnet50,        tv_models.ResNet50_Weights.DEFAULT),
    }

    def __init__(self, num_metadata_features, pretrained=True,
                 cnn_dropout=0.0, mlp_hidden_dims=[128, 64], mlp_dropout=0.3):
        super().__init__()
        self.num_metadata_features = num_metadata_features
        self.image_backbone, self.cnn_dropout, num_image_features = self._build_image_branch(pretrained, cnn_dropout)
        self.metadata_mlp = self._build_metadata_branch(num_metadata_features, mlp_hidden_dims, mlp_dropout)
        self.final_classifier = self._build_classifier(num_image_features, mlp_hidden_dims[-1])

    def _build_image_branch(self, pretrained, cnn_dropout):
        """Loads the torchvision backbone and strips its classifier head."""
        if Config.get_model_config()['architecture'] not in self._BACKBONES:
            raise ValueError(f"Unsupported architecture '{Config.get_model_config()['architecture']}'. Supported: {list(self._BACKBONES)}")
        fn, weights = self._BACKBONES[Config.get_model_config()['architecture']]
        backbone = fn(weights=weights if pretrained else None)
        if hasattr(backbone, 'fc'):            # ResNet: single Linear head
            num_features = backbone.fc.in_features
            backbone.fc = nn.Identity()
        elif hasattr(backbone, 'classifier'):  # EfficientNet / DenseNet: Sequential or Linear head
            head = backbone.classifier
            num_features = (head[-1] if isinstance(head, nn.Sequential) else head).in_features
            backbone.classifier = nn.Identity()
        return backbone, nn.Dropout(cnn_dropout), num_features  # Dropout(p=0) is a no-op

    def _build_metadata_branch(self, num_metadata_features, hidden_dims, dropout):
        """Builds the metadata MLP using torchvision.ops.MLP (Linear + BN + ReLU + Dropout)."""
        return MLP(
            in_channels=num_metadata_features,
            hidden_channels=hidden_dims,
            norm_layer=nn.BatchNorm1d,
            activation_layer=nn.ReLU,
            dropout=dropout,
        )

    def _build_classifier(self, num_image_features, num_metadata_features):
        """Builds the final linear head that fuses image and metadata features."""
        return nn.Linear(num_image_features + num_metadata_features, Config.get_model_config()['num_classes'])

    def forward(self, image_input, metadata_input):
        image_features = self.cnn_dropout(self.image_backbone(image_input))
        metadata_features = self.metadata_mlp(metadata_input)
        return self.final_classifier(torch.cat((image_features, metadata_features), dim=1))

    # ------------------------------------------------------------------ #
    # Class-level factories                                                #
    # ------------------------------------------------------------------ #
    @classmethod
    def build(cls, num_metadata_features: int) -> "MetadataMelanomaModel":
        """Instantiate and move model to the configured device."""
        logger.info("Creating model '%s' with %d metadata features.", Config.get_model_config()['architecture'], num_metadata_features)
        return cls(num_metadata_features=num_metadata_features).to(Config.get_training_config()['device'])

    @staticmethod
    def get_criterion():
        """Focal loss criterion as a partial — handles class imbalance (~13.6% malignant)."""
        loss_cfg = Config.get_loss_config()
        return partial(sigmoid_focal_loss,
                       alpha=loss_cfg['alpha'],
                       gamma=loss_cfg['gamma'],
                       reduction=loss_cfg['reduction'])

    @staticmethod
    def get_optimizer(model: nn.Module, learning_rate: float | None = None) -> torch.optim.Optimizer:
        return torch.optim.Adam(
            model.parameters(),
            lr=learning_rate or Config.get_training_config()['learning_rate'],
        )