import json
import logging
import os
import pickle
import torch
import torch.nn as nn
from pathlib import Path
from huggingface_hub import hf_hub_download

logger = logging.getLogger(__name__)


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

    _GRADCAM_FILENAME            = "gradcam.pth"
    _PREPROCESSOR_FILENAME       = "preprocessor.pkl"
    _METRICS_FILENAME            = "metrics_history.json"
    _ROC_FILENAME                = "roc_curve.png"
    _CONFUSION_FILENAME          = "confusion_matrix.png"
    _SHAP_FILENAME               = "shap_feature_importance.png"
    _OOD_STATS_FILENAME          = "ood_feature_stats.pt"

    _HF_REPO: str | None = None

    @classmethod
    def _get_hf_repo(cls) -> str | None:
        if cls._HF_REPO is None:
            from config import Config
            cls._HF_REPO = Config.get_paths_config().get('hf_model_repo')
        return cls._HF_REPO

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

    def _ensure_from_hf(self, subdir: str, filename: str) -> None:
        """Download *subdir/filename* from the HF repo into the local tree if missing."""
        local = self._root / subdir / filename
        if local.exists():
            return
        repo = self._get_hf_repo()
        if not repo:
            return
        model_name = self._root.name
        remote_path = f"{model_name}/{subdir}/{filename}"
        try:
            logger.info("Downloading %s from %s ...", remote_path, repo)
            downloaded = hf_hub_download(
                repo_id=repo, filename=remote_path, repo_type="model",
            )
            local.parent.mkdir(parents=True, exist_ok=True)
            # Symlink to HF cache to avoid duplication
            local.symlink_to(downloaded)
            logger.info("Cached %s -> %s", local, downloaded)
        except Exception as e:
            logger.warning("HF download failed for %s: %s", remote_path, e)

    def load_preprocessor(self):
        """Load and return the fitted MetadataPreprocessor (HF then local)."""
        self._ensure_from_hf(self._WEIGHTS_SUBDIR, self._PREPROCESSOR_FILENAME)
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

    def shap_plot_path(self) -> Path:
        """SHAP feature importance plot path."""
        return self._plots / self._SHAP_FILENAME

    def ood_stats_path(self) -> Path:
        """OOD feature statistics path."""
        return self._weights / self._OOD_STATS_FILENAME

    def save_ood_stats(self, mean: torch.Tensor, cov_inv: torch.Tensor, threshold: float) -> Path:
        """Save feature mean, inverse covariance, and empirical threshold for Mahalanobis OOD detection."""
        path = self.ood_stats_path()
        torch.save({'mean': mean.cpu(), 'cov_inv': cov_inv.cpu(), 'threshold': threshold}, path)
        logger.info("OOD feature stats saved to %s (threshold=%.1f)", path, threshold)
        return path

    def load_ood_stats(self, map_location=None) -> dict:
        """Load OOD feature stats (HF fallback)."""
        self._ensure_from_hf(self._WEIGHTS_SUBDIR, self._OOD_STATS_FILENAME)
        return torch.load(self.ood_stats_path(), weights_only=True, map_location=map_location)

    @classmethod
    def list_available_runs(cls) -> list[str]:
        """Return names of output dirs that contain a loadable checkpoint.

        Falls back to querying the HF model repo when no local runs exist
        (e.g. on Hugging Face Spaces where weights are downloaded on demand).
        """
        runs: list[str] = []
        if cls._OUTPUT_ROOT.exists():
            for d in sorted(cls._OUTPUT_ROOT.iterdir()):
                if d.is_dir() and (d / cls._WEIGHTS_SUBDIR / cls._GRADCAM_FILENAME).exists():
                    runs.append(d.name)
        if not runs:
            runs = cls._list_runs_from_hf()
        return runs

    @classmethod
    def _list_runs_from_hf(cls) -> list[str]:
        """Discover run names by listing the HF model repo."""
        repo = cls._get_hf_repo()
        if not repo:
            return []
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            files = api.list_repo_files(repo, repo_type="model")
            runs: set[str] = set()
            target = f"/{cls._WEIGHTS_SUBDIR}/{cls._GRADCAM_FILENAME}"
            for f in files:
                if f.endswith(target):
                    runs.add(f.split("/")[0])
            return sorted(runs)
        except Exception as e:
            logger.warning("Could not list runs from HF repo %s: %s", repo, e)
            return []

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

    def save_gradcam_checkpoint(self, model: nn.Module) -> Path:
        """Save model state dict as the stable gradcam.pth inference pointer."""
        path = self.gradcam_checkpoint_path()
        torch.save(model.state_dict(), path)
        return path

    def load_checkpoint(self, model: nn.Module, path: Path | str,
                        map_location: str | torch.device | None = None) -> nn.Module:
        """Load state dict into *model* in-place and return it."""
        state = torch.load(str(path), weights_only=True, map_location=map_location)
        model.load_state_dict(state)
        return model

    def best_available_checkpoint(self) -> Path:
        """Return gradcam.pth if it exists, else the most-recently-modified .pth (excluding preprocessor)."""
        gradcam = self.gradcam_checkpoint_path()
        if gradcam.exists():
            return gradcam
        candidates = [
            p for p in self._weights.glob('*.pth')
            if p.name != self._GRADCAM_FILENAME
        ]
        if not candidates:
            raise FileNotFoundError(f"No checkpoint found in {self._weights}")
        return max(candidates, key=lambda p: p.stat().st_mtime)

    def load_gradcam_checkpoint(self, model: nn.Module,
                                map_location: str | torch.device | None = None) -> nn.Module:
        """Load the best available checkpoint into *model* in-place (HF then local)."""
        self._ensure_from_hf(self._WEIGHTS_SUBDIR, self._GRADCAM_FILENAME)
        return self.load_checkpoint(model, self.best_available_checkpoint(), map_location=map_location)

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
