import logging
import torch
import numpy as np
from sklearn.metrics import roc_curve, auc, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import torchvision.transforms.functional as TF

from config import Config
from file_io_manager import FileIOManager

logger = logging.getLogger(__name__)


class Evaluator:
    def __init__(self, model, criterion, io: FileIOManager | None = None):
        self.model = model
        self.criterion = criterion
        self._device = Config.get_training_config()['device']
        self._io = io

    def evaluate(self, val_loader, use_tta=False):
        self.model.eval()
        total_loss = 0.0
        all_preds: list[int]   = []
        all_labels: list[int]  = []
        all_probs: list[float] = []

        with torch.no_grad():
            for images_batch_tensor, metadata_batch, labels_batch in tqdm(val_loader, desc="Evaluating"):
                metadata_batch = metadata_batch.to(self._device)
                all_labels.extend(labels_batch.cpu().numpy().flatten())

                for i in range(images_batch_tensor.size(0)):
                    img_tensor   = images_batch_tensor[i]
                    meta_single  = metadata_batch[i:i+1]
                    label_single = labels_batch[i:i+1].unsqueeze(1).float().to(self._device)

                    outputs = self.model(img_tensor.unsqueeze(0).to(self._device), meta_single)
                    loss    = self.criterion(outputs, label_single)
                    total_loss += loss.item()

                    if use_tta:
                        probs_tta = [torch.sigmoid(outputs).item()]
                        hflip_out = self.model(TF.hflip(img_tensor).unsqueeze(0).to(self._device), meta_single)
                        probs_tta.append(torch.sigmoid(hflip_out).item())
                        final_prob = float(np.mean(probs_tta))
                    else:
                        final_prob = torch.sigmoid(outputs).item()

                    all_probs.append(final_prob)
                    all_preds.append(1 if final_prob > 0.5 else 0)

        avg_loss = total_loss / len(all_labels) if all_labels else 0.0
        return avg_loss, all_preds, all_labels, all_probs

    def plot_roc_curve(self, labels, probs):
        fpr, tpr, _ = roc_curve(labels, probs)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        path = self._io.roc_curve_path() if self._io else FileIOManager.for_run("default").roc_curve_path()
        plt.savefig(path)
        plt.close()
        logger.info("ROC curve saved to %s", path)

    def plot_confusion_matrix(self, labels, preds):
        cm = confusion_matrix(labels, preds)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Benign', 'Malignant'],
                    yticklabels=['Benign', 'Malignant'])
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix')
        path = self._io.confusion_matrix_path() if self._io else FileIOManager.for_run("default").confusion_matrix_path()
        plt.savefig(path)
        plt.close()
        logger.info("Confusion matrix saved to %s", path)

    def plot_shap(self, val_loader, feature_names: list[str]) -> None:
        """Generate SHAP feature importance plot for the metadata branch.

        Uses KernelExplainer (model-agnostic) to attribute the model's output
        to each of the 14 metadata features while holding image features fixed
        at their validation-set mean.
        """
        import shap

        self.model.eval()

        # Collect metadata + compute mean image features across validation set
        all_metadata: list = []
        all_img_feats: list = []
        with torch.no_grad():
            for img_batch, metadata_batch, _ in val_loader:
                all_metadata.append(metadata_batch)
                img_feats = self.model.cnn_dropout(
                    self.model.image_backbone(img_batch.to(self._device))
                )
                all_img_feats.append(img_feats)
        all_metadata_np = torch.cat(all_metadata, dim=0).cpu().numpy()
        mean_img_feats = torch.cat(all_img_feats, dim=0).mean(dim=0).unsqueeze(0)

        background = shap.kmeans(all_metadata_np, 25)
        test_sample = all_metadata_np[:200]

        explainer = shap.KernelExplainer(
            lambda m: self.model.predict_metadata_proba(m, mean_img_feats),
            background,
        )
        shap_values = explainer.shap_values(test_sample, silent=True)

        plt.figure()
        shap.summary_plot(
            shap_values,
            features=test_sample,
            feature_names=feature_names,
            show=False,
        )
        io = self._io or FileIOManager.for_run("default")
        path = io.shap_plot_path()
        plt.savefig(path, bbox_inches='tight')
        plt.close()
        logger.info("SHAP feature importance plot saved to %s", path)

    def compute_ood_stats(self, val_loader) -> None:
        """Compute mean + inverse covariance of backbone features for Mahalanobis OOD detection.

        Based on: Lee et al., "A Simple Unified Framework for Detecting
        Out-of-Distribution Samples and Adversarial Attacks", NeurIPS 2018.
        https://arxiv.org/abs/1807.03888
        """
        self.model.eval()
        features_list: list[torch.Tensor] = []
        with torch.no_grad():
            for images, _, _ in tqdm(val_loader, desc="Computing OOD stats"):
                images = images.to(self._device)
                feats = self.model.cnn_dropout(self.model.image_backbone(images))
                features_list.append(feats.cpu())

        features = torch.cat(features_list, dim=0).float()
        mean = features.mean(dim=0)
        centered = features - mean
        cov = (centered.T @ centered) / (centered.shape[0] - 1)
        cov += 1e-5 * torch.eye(cov.shape[0])  # regularise for invertibility
        cov_inv = torch.linalg.inv(cov)

        # Compute empirical threshold from validation distances
        dists = (centered @ cov_inv * centered).sum(dim=1)
        threshold = float(dists.mean() + 3 * dists.std())
        logger.info("OOD distances — mean: %.1f, std: %.1f, threshold (μ+3σ): %.1f",
                    dists.mean(), dists.std(), threshold)

        io = self._io or FileIOManager.for_run("default")
        io.save_ood_stats(mean, cov_inv, threshold)
