# Melanoma Classification

EfficientNet/DenseNet/ResNet backbone fused with tabular patient metadata, trained with focal loss to handle class imbalance. Includes a Gradio inference app with EigenCAM explainability.

---

## Setup

```bash
uv sync
```

---

## Usage

All entry points go through `main.py`. Either `--train` or `--app` is required.

```
uv run main.py --train | --app [options]
```

---

## Mode flags

| Flag | Description |
|------|-------------|
| `--train` | Run the training loop |
| `--app` | Launch the Gradio inference app |

---

## All options

### Model

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--architecture` | str | `efficientnet_b0` | Backbone: `efficientnet_b0`, `densenet121`, `resnet50` |
| `--image-size` | int | `256` | Input image size (square) |

### Training

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--lr` | float | `1e-4` | Learning rate |
| `--batch-size` | int | `32` | Batch size |
| `--epochs` | int | `20` | Number of training epochs |
| `--device` | str | auto | Compute device: `cuda` or `cpu` |
| `--num-workers` | int | `4` | DataLoader worker processes |

### Data paths

| Flag | Type | Description |
|------|------|-------------|
| `--data-dir` | str | Directory containing training images |
| `--labels-csv` | str | Path to training labels CSV |

### Evaluation

| Flag | Description |
|------|-------------|
| `--tta` | Enable test-time augmentation during evaluation |

### App

| Flag | Description |
|------|-------------|
| `--share` | Create a public Gradio share link (`--app` only) |

---

## Examples

```bash
# Train with all defaults
uv run main.py --train

# Train with a different backbone and more epochs
uv run main.py --train --architecture densenet121 --epochs 40

# Train on a custom dataset with a higher learning rate on GPU
uv run main.py --train \
  --data-dir /data/melanoma/train \
  --labels-csv /data/melanoma/labels.csv \
  --lr 3e-4 --batch-size 64 --device cuda

# Train with ResNet-50, smaller images, TTA enabled during validation
uv run main.py --train --architecture resnet50 --image-size 224 --tta

# Smoke test — 1 epoch, 0 workers: verifies the full pipeline
# (data loading → forward pass → loss → checkpoint → metrics JSON → plots)
# Use --batch-size 64 on CUDA; keep 32 on CPU/MPS to avoid OOM
uv run main.py --train --epochs 1 --batch-size 32 --num-workers 0

# Launch the inference app locally
uv run main.py --app

# Launch the inference app with a public share link
uv run main.py --app --share

# Launch the app and force CPU inference
uv run main.py --app --device cpu
```

---

## Output layout

```
output/
  {architecture}/
    weights/    # best_ep{n}.pth  +  gradcam.pth
    metrics/    # metrics_history.json
    plots/      # roc_curve.png  +  confusion_matrix.png
```

---

## Pushing models to Hugging Face

Trained weights are hosted at [huggingface.co/burrows99/melanoma-models](https://huggingface.co/burrows99/melanoma-models).

```bash
# Install the Hugging Face CLI (included in huggingface_hub)
uv pip install huggingface_hub

# Login (opens browser or prompts for token)
hf auth login

# Upload the entire output folder to the repo
hf upload burrows99/melanoma-models ./output . --repo-type model
```
