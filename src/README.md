# Melanoma Detection Project

Deep learning model for melanoma detection from skin lesion images, incorporating patient metadata and Grad-CAM for explainability.

## Setup

1.  **Environment:** Create a Python virtual environment and install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
   *( PyTorch nightly build for 12.8 CUDA is used.)*

2.  **Model Weights for `app.py`:**
    The Gradio application (`app.py`) expects pre-trained weights at `result/weights/gradcam.pth`. If training a new model, update this path in `app.py` or rename the saved model accordingly.

## Running the Application

To launch the Gradio interface for inference and Grad-CAM visualization:
```bash
python app.py
```

## Training the Model

1.  **Dataset:**
    Configure `TRAIN_DATA_DIR` (image folder) and `TRAIN_LABELS_PATH` (labels CSV) in `config.py`.

2.  **Configuration:**
    Adjust training parameters (model architecture, learning rate, etc.) in `config.py`.

3.  **Weights & Biases (W&B):**
    Training uses W&B for logging. You will be prompted to log in.
    To disable W&B, set the environment variable `WANDB_MODE=disabled` before running but will limits result output:
    ```bash
    # $env:WANDB_MODE="disabled"; python train.py
    ```

4.  **Run Training:**
    ```bash
    python train.py
    ```

## Key Files

*   `app.py`: Gradio application.
*   `train.py`: Model training script.
*   `evaluate.py`: Evaluation logic (metrics, TTA).
*   `dataset.py`: Data loading and augmentations.
*   `model.py`: Neural network definition.
*   `config.py`: Project configurations.
*   `result`: path where the weights are stored
*   `base`: path where the weights are stored





#