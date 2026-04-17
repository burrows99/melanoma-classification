import logging
import torch
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import gradio as gr

from model import MetadataMelanomaModel
from config import Config
from dataset import Transform
from file_io_manager import FileIOManager
from typing import Any, cast
from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

logger = logging.getLogger(__name__)


class App:
    def __init__(self):
        self._device        = Config.get_training_config()['device']
        self._transform     = Transform(train=False)
        self._io            : FileIOManager | None            = None
        self._preprocessor  : Any                            = None
        self._model         : MetadataMelanomaModel | None   = None
        self._target_layers : dict                           = {}

    def load_model(self) -> str:
        """Load preprocessor + model; return status text."""
        try:
            io           = FileIOManager.for_run(Config.MODEL_NAME)
            preprocessor = io.load_preprocessor()
            model        = MetadataMelanomaModel.build(
                num_metadata_features=preprocessor.num_output_features
            )
            io.load_gradcam_checkpoint(model, map_location=self._device)
            model.to(self._device).eval()

            self._io            = io
            self._preprocessor  = preprocessor
            self._model         = model
            self._target_layers = self._get_target_layers()
            logger.info("Model loaded successfully.")
            return "Model loaded successfully."
        except Exception as e:
            logger.exception("Error loading model")
            return f"Error loading model: {e}"

    def _get_target_layers(self) -> dict:
        assert self._model is not None
        bb = self._model.image_backbone
        layers: dict = {}
        if hasattr(bb, 'conv_head'):
            layers['conv_head'] = bb.conv_head
        if hasattr(bb, 'blocks'):
            n = len(bb.blocks)
            for label, idx in [('blocks[-3].conv_pwl', n - 3), (f'blocks[{n // 2}].conv_pwl', n // 2)]:
                if idx >= 0 and hasattr(bb.blocks[idx], 'conv_pwl'):
                    layers[label] = bb.blocks[idx].conv_pwl
        if not layers:
            for name, mod in reversed(list(bb.named_modules())):
                if isinstance(mod, torch.nn.Conv2d):
                    layers[name] = mod
                    break
        logger.info("CAM layers: %s", list(layers))
        return layers

    def _prepare_metadata(self, age=None, sex=None, site=None) -> torch.Tensor:
        defaults = Config.get_metadata_config()['defaults']
        row = {
            'age_approx':                       float(age) if age is not None else defaults['age_approx'],
            'sex':                              sex  if sex  else defaults['sex'],
            'anatom_site_general_challenge':    site if site else defaults['anatom_site_general_challenge'],
        }
        arr = self._preprocessor.transform(pd.DataFrame([row]))
        return torch.tensor(arr, dtype=torch.float32).to(self._device)

    def predict_and_visualize(self, img, age=None, sex='male', site='torso'):
        if img is None:
            return "Please upload an image", None

        if self._model is None:
            return "No model loaded. Select a model from the dropdown.", None

        try:
            img_rgb, img_resized, img_tensor = self._preprocess_image(img)
            metadata_tensor = self._prepare_metadata(age, sex, site)
            prediction_text = self._run_tta(img_rgb, metadata_tensor)

            if prediction_text is None:
                return "Error: TTA failed for all augmentations.", None

            default_layer = self._target_layers.get('blocks[-3].conv_pwl') and 'blocks[-3].conv_pwl' or next(iter(self._target_layers))
            _, side_by_side = self._run_eigencam(img_resized, img_tensor, default_layer)
            return prediction_text, side_by_side

        except Exception:
            logger.exception("Prediction failed")
            return "Error", None

    def _preprocess_image(self, img) -> tuple:
        img_rgb     = img.convert('RGB')
        size        = Config.get_model_config()['image_size']
        img_resized = img_rgb.resize((size, size), Image.Resampling.LANCZOS)
        img_tensor  = self._transform(img_resized).unsqueeze(0).to(self._device)
        return img_rgb, img_resized, img_tensor

    def _run_tta(self, img_rgb, metadata_tensor) -> str | None:
        assert self._model is not None
        probs: list[float] = []
        logger.info("Starting TTA with %d augmentations...", len(Transform.tta_transforms))
        for aug_name, tta_fn in Transform.tta_transforms.items():
            try:
                aug_tensor = self._transform(tta_fn(img_rgb)).unsqueeze(0).to(self._device)
                with torch.no_grad():
                    prob = torch.sigmoid(self._model(aug_tensor, metadata_tensor)).item()
                probs.append(prob)
                logger.debug("  TTA - %s: %.4f", aug_name, prob)
            except Exception as e:
                logger.warning("  TTA %s failed: %s", aug_name, e)

        if not probs:
            return None

        final_prob = float(np.mean(probs))
        logger.info("TTA averaged probability: %.4f over %d augmentations", final_prob, len(probs))
        benign = (1 - final_prob) * 100
        malignant = final_prob * 100
        return f"Benign:     {benign:.1f}%\nMalignant:  {malignant:.1f}%"

    def _run_eigencam(self, img_resized, img_tensor, target_layer_name: str) -> tuple:
        assert self._model is not None
        if target_layer_name not in self._target_layers:
            target_layer_name = next(iter(self._target_layers))
            logger.warning("Layer not found; falling back to '%s'", target_layer_name)
        try:
            cam       = EigenCAM(model=self._model.image_backbone, target_layers=[self._target_layers[target_layer_name]])
            grayscale = cam(input_tensor=img_tensor, targets=cast(Any, [ClassifierOutputTarget(0)]))[0]
            lo, hi    = grayscale.min(), grayscale.max()
            grayscale = (grayscale - lo) / (hi - lo) if hi != lo else grayscale
            rgb_f     = np.array(img_resized).astype(np.float32) / 255.0
            cam_img   = (show_cam_on_image(rgb_f, grayscale, use_rgb=True, colormap=cv2.COLORMAP_JET, image_weight=0.5) * 255).astype(np.uint8)
            return cam_img, np.hstack((np.array(img_resized), cam_img))
        except Exception:
            logger.exception("EigenCAM failed")
            return None, None


    def build_interface(self) -> gr.Blocks:
        with gr.Blocks(title="Melanoma Detection") as iface:
            gr.Markdown("## Melanoma Detection")

            # ── Model status row ─────────────────────────────────────────
            with gr.Row():
                model_status = gr.Textbox(
                    label="Status", interactive=False,
                )

            # ── Main content ─────────────────────────────────────────────
            with gr.Row():

                # Left: image + metadata
                with gr.Column(scale=1):
                    image_input = gr.Image(type="pil", label="Skin Lesion Image")

                    with gr.Group():
                        gr.Markdown("**Patient Metadata**")
                        age_input  = gr.Number(label="Age", value=50)
                        sex_input  = gr.Dropdown(
                            label="Sex", choices=["male", "female", "unknown"], value="male"
                        )
                        site_input = gr.Dropdown(
                            label="Anatomical Site",
                            choices=["torso", "lower extremity", "upper extremity",
                                     "head/neck", "palms/soles", "oral/genital",
                                     "anterior torso", "posterior torso",
                                     "lateral torso", "unknown"],
                            value="torso",
                        )
                    predict_btn = gr.Button("Analyse", variant="primary")

                # Right: prediction + heatmap
                with gr.Column(scale=2):
                    prediction_output = gr.Textbox(
                        label="Prediction", lines=2, interactive=False,
                    )
                    heatmap_output = gr.Image(
                        label="Original vs. Heatmap",
                    )

            # ── Events ───────────────────────────────────────────────────
            iface.load(
                fn=self.load_model,
                outputs=[model_status],
            )

            predict_btn.click(
                fn=self.predict_and_visualize,
                inputs=[image_input, age_input, sex_input, site_input],
                outputs=[prediction_output, heatmap_output],
            )

        return iface

    def launch(self, **kwargs) -> None:
        self.build_interface().launch(**kwargs)
