---
title: Melanoma Classification
emoji: 🔬
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: "6.12.0"
python_version: "3.12"
app_file: app.py
models:
  - burrows99/melanoma-models
preload_from_hub:
  - burrows99/melanoma-models
short_description: Gradio app to detect melanoma with EigenCAM explainability
license: apache-2.0
---

# Melanoma Classification

EfficientNet-B0 backbone fused with tabular patient metadata, trained with focal loss to handle class imbalance. Includes a Gradio inference app with EigenCAM explainability.

---

## Setup

```bash
uv sync
```

---

## Usage

All entry points go through `main.py`. Either `--train` or `--app` is required.

```
uv run main.py --train | --app  [options]
```

---

## Mode flags

| Flag | Description |
|------|-------------|
| `--train` | Run the training loop |
| `--app` | Launch the Gradio inference app |

---

## All options

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

# Train on a custom dataset with a higher learning rate on GPU
uv run main.py --train \
  --data-dir /data/melanoma/train \
  --labels-csv /data/melanoma/labels.csv \
  --lr 3e-4 --batch-size 64 --device cuda

# Train with TTA enabled during validation
uv run main.py --train --tta

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
  efficientnet_b0/
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
