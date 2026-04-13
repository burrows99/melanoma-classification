import json
import os
import pickle
import torch
import torch.nn as nn
from pathlib import Path


class FileIOManager:
    """Centralizes all path construction and I/O for a single output tree.

    Layout::

        output/
          {model_name}/
            weights/   — training checkpoints + gradcam.pth
            metrics/   — per-epoch JSON history  (metrics_history.json)
            plots/     — roc_curve.png, confusion_matrix.png

    Obtain an instance via ``FileIOManager.for_run(model_name)``.
    The static helper ``image_path`` needs no instance.
    """

    _OUTPUT_ROOT     = Path("output")
    _WEIGHTS_SUBDIR  = "weights"
    _METRICS_SUBDIR  = "metrics"
    _PLOTS_SUBDIR    = "plots"

    _GRADCAM_FILENAME       = "gradcam.pth"
    _PREPROCESSOR_FILENAME  = "preprocessor.pkl"
    _METRICS_FILENAME       = "metrics_history.json"
    _ROC_FILENAME           = "roc_curve.png"
    _CONFUSION_FILENAME     = "confusion_matrix.png"

    def __init__(self, model_name: str) -> None:
        self._root    = self._OUTPUT_ROOT / model_name
        self._weights = self._root / self._WEIGHTS_SUBDIR
        self._metrics = self._root / self._METRICS_SUBDIR
        self._plots   = self._root / self._PLOTS_SUBDIR
        for d in (self._weights, self._metrics, self._plots):
            d.mkdir(parents=True, exist_ok=True)

    @classmethod
    def for_run(cls, model_name: str) -> "FileIOManager":
        """Factory: return a manager bound to *model_name*."""
        return cls(model_name)

    # ------------------------------------------------------------------ #
    # Path constructors                                                    #
    # ------------------------------------------------------------------ #
    def checkpoint_path(self, run_name: str, epoch: int) -> Path:
        """Per-epoch best-model checkpoint path."""
        return self._weights / f"{run_name}_best_ep{epoch}.pth"

    def gradcam_checkpoint_path(self) -> Path:
        """Fixed GradCAM inference checkpoint path."""
        return self._weights / self._GRADCAM_FILENAME

    def preprocessor_path(self) -> Path:
        """Fitted MetadataPreprocessor pickle path."""
        return self._weights / self._PREPROCESSOR_FILENAME

    def save_preprocessor(self, preprocessor) -> Path:
        """Pickle the fitted MetadataPreprocessor; return the path written."""
        path = self.preprocessor_path()
        with open(path, 'wb') as f:
            pickle.dump(preprocessor, f)
        return path

    def load_preprocessor(self):
        """Load and return the fitted MetadataPreprocessor."""
        with open(self.preprocessor_path(), 'rb') as f:
            return pickle.load(f)

    def metrics_path(self) -> Path:
        """JSON metrics history file path."""
        return self._metrics / self._METRICS_FILENAME

    def roc_curve_path(self) -> Path:
        """ROC curve plot path."""
        return self._plots / self._ROC_FILENAME

    def confusion_matrix_path(self) -> Path:
        """Confusion matrix plot path."""
        return self._plots / self._CONFUSION_FILENAME

    @staticmethod
    def image_path(data_dir: str, image_name: str) -> str:
        """Full path to a dataset image (no model-binding needed)."""
        return os.path.join(data_dir, f"{image_name}.jpg")

    # ------------------------------------------------------------------ #
    # Checkpoint I/O                                                       #
    # ------------------------------------------------------------------ #
    def save_checkpoint(self, model: nn.Module, run_name: str, epoch: int) -> Path:
        """Save model state dict; return the path written."""
        path = self.checkpoint_path(run_name, epoch)
        torch.save(model.state_dict(), path)
        return path

    def load_checkpoint(self, model: nn.Module, path: Path | str,
                        map_location: str | torch.device | None = None) -> nn.Module:
        """Load state dict into *model* in-place and return it."""
        state = torch.load(str(path), weights_only=True, map_location=map_location)
        model.load_state_dict(state)
        return model

    def load_gradcam_checkpoint(self, model: nn.Module,
                                map_location: str | torch.device | None = None) -> nn.Module:
        """Load the fixed GradCAM checkpoint into *model* in-place."""
        return self.load_checkpoint(model, self.gradcam_checkpoint_path(), map_location=map_location)

    # ------------------------------------------------------------------ #
    # Metrics I/O                                                          #
    # ------------------------------------------------------------------ #
    def append_epoch_metrics(self, metrics: dict) -> None:
        """Append one epoch's metrics dict to the JSON history file."""
        path = self.metrics_path()
        history: list[dict] = []
        if path.exists():
            with open(path) as f:
                history = json.load(f)
        history.append(metrics)
        with open(path, "w") as f:
            json.dump(history, f, indent=2)
