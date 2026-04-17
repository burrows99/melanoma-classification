import json
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

    @staticmethod
    def _load_best_metrics(output_root) -> list[tuple[str, dict]]:
        entries: list[tuple[str, dict]] = []
        for model_dir in sorted(output_root.iterdir()):
            if 'smoke_test' in model_dir.name:
                continue
            metrics_file = model_dir / FileIOManager._METRICS_SUBDIR / FileIOManager._METRICS_FILENAME
            if metrics_file.exists():
                with open(metrics_file) as f:
                    history = json.load(f)
                best = {k: max(e[k] for e in history) for k in ('val_f1', 'val_acc', 'val_recall')}
                entries.append((model_dir.name, best))
        return entries

    @staticmethod
    def _plot_best_metrics_bar(ax, entries: list[tuple[str, dict]]) -> None:
        metrics_keys = ['val_f1', 'val_acc', 'val_recall']
        metric_labels = ['F1', 'Accuracy', 'Recall']
        n_models = len(entries)
        bar_width = 0.2
        x = np.arange(len(metrics_keys))
        for i, (model_name, best) in enumerate(entries):
            offset = (i - (n_models - 1) / 2) * bar_width
            ax.bar(x + offset, [best[k] for k in metrics_keys], width=bar_width, label=model_name)
        ax.set_xticks(x)
        ax.set_xticklabels(metric_labels)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel('Score')
        ax.set_title('Best Validation Metrics')
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        ax.grid(True, axis='y', alpha=0.3)

    @staticmethod
    def plot_metrics_comparison() -> None:
        """Grouped bar chart of best val metrics for all trained models.
        Reads metrics_history.json; saves output/metrics_comparison.png.
        """
        entries = Evaluator._load_best_metrics(FileIOManager._OUTPUT_ROOT)
        if not entries:
            logger.warning("No metrics_history.json files found under %s — run training first.", FileIOManager._OUTPUT_ROOT)
            return

        fig, ax = plt.subplots(figsize=(8, 5))
        Evaluator._plot_best_metrics_bar(ax, entries)
        fig.tight_layout()
        out = FileIOManager.metrics_comparison_path()
        fig.savefig(out)
        plt.close(fig)
        logger.info("Metrics comparison saved to %s", out)

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
