import logging
import numpy as np
import torch
import torch.nn as nn
from torchvision.ops import MLP, sigmoid_focal_loss
from functools import partial
from config import Config

logger = logging.getLogger(__name__)


# --- Metadata Fusion Model ---
class MetadataMelanomaModel(nn.Module):
    """Model combining image features (CNN backbone) and tabular metadata (MLP)."""

    def __init__(self, num_metadata_features, pretrained=True,
                 cnn_dropout=0.0, mlp_hidden_dims=[128, 64], mlp_dropout=0.3):
        super().__init__()
        self.num_metadata_features = num_metadata_features
        self.image_backbone, self.cnn_dropout, num_image_features = self._build_image_branch(pretrained, cnn_dropout)

        exp = Config.get_experiment_config()
        self._image_only = bool(exp and exp.get('image_only'))

        if self._image_only:
            self.metadata_mlp = nn.Identity()
            self.final_classifier = self._build_classifier(num_image_features, 0)
        else:
            self.metadata_mlp = self._build_metadata_branch(num_metadata_features, mlp_hidden_dims, mlp_dropout)
            self.final_classifier = self._build_classifier(num_image_features, mlp_hidden_dims[-1])

    def _build_image_branch(self, pretrained, cnn_dropout):
        """Loads the backbone from Config and strips its classifier head."""
        backbone = Config.BACKBONE_FN(
            weights=Config.BACKBONE_WEIGHTS if pretrained else None
        )
        head = backbone.classifier
        num_features = (head[-1] if isinstance(head, nn.Sequential) else head).in_features
        backbone.classifier = nn.Identity()
        return backbone, nn.Dropout(cnn_dropout), num_features

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
        return nn.Linear(num_image_features + num_metadata_features, 1)

    def fuse_and_classify(self, image_features, metadata_input):
        """Fuse pre-extracted image features with metadata and classify."""
        metadata_features = self.metadata_mlp(metadata_input)
        return self.final_classifier(torch.cat((image_features, metadata_features), dim=1))

    @torch.no_grad()
    def predict_metadata_proba(self, metadata_np: np.ndarray, mean_img_feats: torch.Tensor) -> np.ndarray:
        """Predict probabilities from metadata (numpy) with fixed image features. Used by SHAP."""
        meta_t = torch.tensor(metadata_np, dtype=torch.float32, device=mean_img_feats.device)
        img_expanded = mean_img_feats.expand(meta_t.shape[0], -1)
        return torch.sigmoid(self.fuse_and_classify(img_expanded, meta_t)).cpu().numpy().flatten()

    def forward(self, image_input, metadata_input):
        image_features = self.cnn_dropout(self.image_backbone(image_input))
        if self._image_only:
            return self.final_classifier(image_features)
        return self.fuse_and_classify(image_features, metadata_input)

    # ------------------------------------------------------------------ #
    # Class-level factories                                                #
    # ------------------------------------------------------------------ #
    @classmethod
    def build(cls, num_metadata_features: int) -> "MetadataMelanomaModel":
        """Instantiate and move model to the configured device."""
        logger.info("Creating EfficientNet-B0 model with %d metadata features.", num_metadata_features)
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
        lr = learning_rate or Config.get_training_config()['learning_rate']
        exp = Config.get_experiment_config()
        if exp and exp['optimizer'] == 'AdamW':
            return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=exp['weight_decay'])
        return torch.optim.Adam(model.parameters(), lr=lr)

    @staticmethod
    def get_scheduler(optimizer: torch.optim.Optimizer):
        """Return a LR scheduler if the active experiment requires one, else None."""
        exp = Config.get_experiment_config()
        if exp and exp.get('scheduler') == 'CosineAnnealingLR':
            T_max = Config.get_training_config()['num_epochs']
            return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)
        return None