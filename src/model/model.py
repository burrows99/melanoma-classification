import torch
from functools import partial
from torchvision.ops import sigmoid_focal_loss
from config import MODEL_ARCHITECTURE, DEVICE, LEARNING_RATE, LOSS_FUNCTION_TYPE, FOCAL_LOSS_ALPHA, FOCAL_LOSS_GAMMA, FOCAL_LOSS_REDUCTION
from model.metadata_melanoma_model import MetadataMelanomaModel


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

# --- Criterion and Optimizer Functions ---
def get_criterion(loss_type=LOSS_FUNCTION_TYPE):
    # Focal loss is used because the melanoma dataset is heavily class-imbalanced:
    # only ~13.6% of samples are malignant (melanoma) and ~86.4% are benign.
    # Standard cross-entropy would be dominated by the easy benign examples and
    # underfit on the rare malignant class. Focal loss down-weights well-classified
    # easy examples (via the (1-pt)^gamma term) so training focuses on hard,
    # misclassified samples — especially the minority malignant class.
    # alpha=0.864 directly reflects the inverse malignant proportion (1 - 0.136),
    # giving extra weight to positive (malignant) examples in the loss.
    return partial(sigmoid_focal_loss, alpha=FOCAL_LOSS_ALPHA, gamma=FOCAL_LOSS_GAMMA, reduction=FOCAL_LOSS_REDUCTION)

def get_optimizer(model, learning_rate=LEARNING_RATE):
    return torch.optim.Adam(model.parameters(), lr=learning_rate) 