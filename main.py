import argparse
import logging
import sys
from pathlib import Path
import kagglehub

logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent / "src"))

from config import Config  # noqa: E402  (after sys.path patch)


def _ensure_dataset() -> None:
    """Download dataset from Kaggle if dataset/train is absent or empty."""
    data_dir = Path('dataset/train')
    csv_path = Path('dataset/train_concat.csv')

    if data_dir.exists() and csv_path.exists() and any(data_dir.iterdir()):
        return

    logger.info("Dataset not found in CWD — downloading from Kaggle (cached after first run)...")
    src = Path(kagglehub.dataset_download(Config.get_paths_config()['kaggle_dataset']))
    # Layout: <src>/train/train/<images>  +  <src>/train_concat.csv

    Path('dataset').mkdir(exist_ok=True)

    if not data_dir.exists():
        data_dir.symlink_to((src / 'train' / 'train').resolve())
    if not csv_path.exists():
        csv_path.symlink_to((src / 'train_concat.csv').resolve())

    logger.info(f"Dataset ready: {len(list(data_dir.iterdir()))} images, CSV at {csv_path}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Melanoma classification — train a model or launch the Gradio app.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--train", action="store_true", help="Run training loop")
    mode.add_argument("--app",   action="store_true", help="Launch Gradio inference app")
    mode.add_argument("--repair-ood", action="store_true", dest="repair_ood",
                      help="Recompute OOD stats (with empirical threshold) for all saved runs")

    # --- Training config overrides ---
    parser.add_argument("--experiment", type=int, choices=[1, 2, 3, 4],
                        help="Experiment preset (1: CosineAnneal, 2: AdamW+Cosine, 3: AdamW+Cosine+γ1.5, 4: image-only ablation)")
    parser.add_argument("--lr", type=float, dest="learning_rate",
                        help="Learning rate")
    parser.add_argument("--batch-size", type=int, dest="batch_size")
    parser.add_argument("--epochs", type=int, dest="num_epochs")
    parser.add_argument("--device", type=str,
                        help="Compute device, e.g. cuda or cpu")
    parser.add_argument("--num-workers", type=int, dest="num_workers")

    # --- Paths config overrides ---
    parser.add_argument("--data-dir", type=str, dest="train_data_dir",
                        help="Directory containing training images")
    parser.add_argument("--labels-csv", type=str, dest="train_labels_path",
                        help="Path to training labels CSV")

    # --- Evaluation config overrides ---
    parser.add_argument("--tta", action="store_true", default=None, dest="tta_enabled",
                        help="Enable test-time augmentation during evaluation")

    # --- App-only options ---
    parser.add_argument("--share", action="store_true",
                        help="Create a public Gradio share link (--app only)")

    return parser


def _repair_ood() -> None:
    """Recompute OOD stats (with empirical threshold) for every saved run."""
    import re as _re
    from dataset import MelanomaDataLoaders
    from model import MetadataMelanomaModel
    from evaluate import Evaluator
    from file_io_manager import FileIOManager

    runs = FileIOManager.list_available_runs()
    if not runs:
        logger.error("No saved runs found in output/")
        return

    logger.info("Repairing OOD stats for runs: %s", runs)

    # Build val loader once (shared across all runs)
    Config.set_experiment(None)
    loaders = MelanomaDataLoaders()
    val_loader = loaders.get_val_loader()

    for run_name in runs:
        logger.info("── %s ──", run_name)
        m = _re.match(r'^experiment(\d+)$', run_name)
        Config.set_experiment(int(m.group(1)) if m else None)

        io = FileIOManager.for_run(run_name)
        try:
            preprocessor = io.load_preprocessor()
        except Exception:
            logger.warning("  No preprocessor for %s — skipping", run_name)
            continue

        model = MetadataMelanomaModel.build(
            num_metadata_features=preprocessor.num_output_features
        )
        try:
            io.load_gradcam_checkpoint(model, map_location=Config.get_training_config()['device'])
        except Exception:
            logger.warning("  No gradcam.pth for %s — skipping", run_name)
            continue

        model.eval()
        evaluator = Evaluator(model, MetadataMelanomaModel.get_criterion(), io=io)
        evaluator.compute_ood_stats(val_loader)
        logger.info("  ✓ %s done", run_name)

    logger.info("All OOD stats repaired.")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    args = _build_parser().parse_args()

    # Build overrides: only keys with non-None values that aren't mode/share flags
    _mode_keys = {"train", "app", "share", "experiment"}
    overrides = {k: v for k, v in vars(args).items() if k not in _mode_keys and v is not None}
    if overrides:
        Config.override(**overrides)

    if args.train:
        if args.experiment:
            Config.set_experiment(args.experiment)
        _ensure_dataset()
        from train import Trainer
        Trainer().train()
    elif args.repair_ood:
        _ensure_dataset()
        _repair_ood()
    else:
        from app import App
        App().launch(share=args.share)


if __name__ == "__main__":
    main()
