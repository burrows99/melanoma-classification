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
    mode.add_argument("--compare", action="store_true", dest="compare",
                      help="Compare all trained models using metrics_history.json and save output/metrics_comparison.png")

    # --- Model config overrides ---
    parser.add_argument("--architecture", type=str,
                        help="Backbone: efficientnet_b0 | densenet121 | resnet50")
    parser.add_argument("--image-size", type=int, dest="image_size",
                        help="Input image size (square)")

    # --- Training config overrides ---
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


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    args = _build_parser().parse_args()

    # Build overrides: only keys with non-None values that aren't mode/share flags
    _mode_keys = {"train", "app", "compare", "share"}
    overrides = {k: v for k, v in vars(args).items() if k not in _mode_keys and v is not None}
    if overrides:
        Config.override(**overrides)

    if args.train:
        _ensure_dataset()
        from train import Trainer
        Trainer().train()
    elif args.compare:
        from evaluate import Evaluator
        Evaluator.plot_metrics_comparison()
    else:
        from app import App
        App().launch(share=args.share)


if __name__ == "__main__":
    main()
