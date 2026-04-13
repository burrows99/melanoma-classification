import torch
import numpy as np
from sklearn.metrics import roc_curve, auc, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from PIL import Image
import torchvision.transforms.functional as TF

from config import Config
from file_io_manager import FileIOManager


# TTA transforms using PIL — Image.Transpose enum avoids deprecated integer constants
tta_transforms_pil = {
    'original': lambda img: img,
    'hflip':   lambda img: img.transpose(Image.Transpose.FLIP_LEFT_RIGHT),
    'vflip':   lambda img: img.transpose(Image.Transpose.FLIP_TOP_BOTTOM),
    'rot90':   lambda img: img.rotate(90,  expand=False),
    'rot180':  lambda img: img.rotate(180, expand=False),
    'rot270':  lambda img: img.rotate(270, expand=False),
}


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
        print(f"ROC curve saved to {path}")

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
        print(f"Confusion matrix saved to {path}")
