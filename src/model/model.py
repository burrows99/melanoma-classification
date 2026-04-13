import torch
from functools import partial
from torchvision.ops import sigmoid_focal_loss
from config import Config
from model.metadata_melanoma_model import MetadataMelanomaModel


# --- Model Instantiation Function ---
def get_model(num_metadata_features):
    """Instantiates the metadata fusion model.
    Args:
        num_metadata_features (int): Number of features in the metadata input.
    Returns:
        torch.nn.Module: The instantiated model, moved to DEVICE.
    """
    print(f"Creating metadata fusion model '{Config.get_model_config()['architecture']}' with {num_metadata_features} metadata features.")
    model = MetadataMelanomaModel(num_metadata_features=num_metadata_features)
    model = model.to(Config.get_training_config()['device'])
    return model

# --- Criterion and Optimizer Functions ---
def get_criterion(loss_type=None):
    # Focal loss is used because the melanoma dataset is heavily class-imbalanced:
    # only ~13.6% of samples are malignant (melanoma) and ~86.4% are benign.
    # Standard cross-entropy would be dominated by the easy benign examples and
    # underfit on the rare malignant class. Focal loss down-weights well-classified
    # easy examples (via the (1-pt)^gamma term) so training focuses on hard,
    # misclassified samples — especially the minority malignant class.
    # alpha=0.864 directly reflects the inverse malignant proportion (1 - 0.136),
    # giving extra weight to positive (malignant) examples in the loss.
    return partial(sigmoid_focal_loss, alpha=Config.get_loss_config()['alpha'], gamma=Config.get_loss_config()['gamma'], reduction=Config.get_loss_config()['reduction'])

def get_optimizer(model, learning_rate=None):
    learning_rate = learning_rate or Config.get_training_config()['learning_rate']
    return torch.optim.Adam(model.parameters(), lr=learning_rate) 