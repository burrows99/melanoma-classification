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


class App:
    def __init__(self):
        self._device    = Config.get_training_config()['device']
        self._io        = FileIOManager.for_run(Config.get_model_config()['architecture'])
        self._transform = Transform(train=False)

        try:
            self._preprocessor = self._io.load_preprocessor()
            print(f"Preprocessor loaded from {self._io.preprocessor_path()}")
        except Exception as e:
            raise RuntimeError(
                f"Could not load preprocessor from {self._io.preprocessor_path()}. "
                "Run --train first."
            ) from e

        self._model = MetadataMelanomaModel.build(
            num_metadata_features=self._preprocessor.num_output_features
        )
        try:
            self._io.load_gradcam_checkpoint(self._model, map_location=self._device)
            print(f"Model loaded from {self._io.gradcam_checkpoint_path()}")
        except Exception as e:
            print(f"Error loading model: {e}")
        self._model.to(self._device).eval()

        self._target_layers = self._get_target_layers()

    def _get_target_layers(self) -> dict:
        backbone = self._model.image_backbone
        layers: dict = {}

        if hasattr(backbone, 'conv_head'):
            layers['conv_head'] = backbone.conv_head

        if hasattr(backbone, 'blocks'):
            if len(backbone.blocks) >= 3:
                idx = len(backbone.blocks) - 3
                if hasattr(backbone.blocks[idx], 'conv_pwl'):
                    layers['blocks[-3].conv_pwl'] = backbone.blocks[idx].conv_pwl
            if len(backbone.blocks) >= 2:
                mid = len(backbone.blocks) // 2
                if hasattr(backbone.blocks[mid], 'conv_pwl'):
                    layers[f'blocks[{mid}].conv_pwl'] = backbone.blocks[mid].conv_pwl

        if not layers:
            for name, module in reversed(list(backbone.named_modules())):
                if isinstance(module, torch.nn.Conv2d):
                    layers[name] = module
                    break

        print(f"Found {len(layers)} target layers for Grad-CAM:")
        for name in layers:
            print(f"  - {name}")
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

    def predict_and_visualize(self, img, age=None, sex='male', site='torso',
                              target_layer_name='blocks[-3].conv_pwl'):
        if img is None:
            return "Please upload an image", None, None

        try:
            img_rgb, img_resized, img_tensor = self._preprocess_image(img)
            metadata_tensor = self._prepare_metadata(age, sex, site)
            prediction_text = self._run_tta(img_rgb, metadata_tensor)

            if prediction_text is None:
                return "Error: TTA failed for all augmentations.", None, None

            cam_image, side_by_side = self._run_eigencam(img_resized, img_tensor, target_layer_name)
            return prediction_text, cam_image, side_by_side

        except Exception:
            import traceback
            traceback.print_exc()
            return "Error", None, None

    def _preprocess_image(self, img) -> tuple:
        img_rgb     = img.convert('RGB')
        size        = Config.get_model_config()['image_size']
        img_resized = img_rgb.resize((size, size), Image.Resampling.LANCZOS)
        img_tensor  = self._transform(img_resized).unsqueeze(0).to(self._device)
        return img_rgb, img_resized, img_tensor

    def _run_tta(self, img_rgb, metadata_tensor) -> str | None:
        probs: list[float] = []
        print(f"Starting TTA with {len(Transform.tta_transforms)} augmentations...")
        for aug_name, tta_fn in Transform.tta_transforms.items():
            try:
                aug_tensor = self._transform(tta_fn(img_rgb)).unsqueeze(0).to(self._device)
                with torch.no_grad():
                    prob = torch.sigmoid(self._model(aug_tensor, metadata_tensor)).item()
                probs.append(prob)
                print(f"  TTA - {aug_name}: {prob:.4f}")
            except Exception as e:
                print(f"  TTA {aug_name} failed: {e}")

        if not probs:
            return None

        final_prob = float(np.mean(probs))
        print(f"TTA averaged probability: {final_prob:.4f} over {len(probs)} augmentations")
        return f"Melanoma probability (TTA): {final_prob:.4f}"

    def _run_eigencam(self, img_resized, img_tensor, target_layer_name: str) -> tuple:
        if target_layer_name not in self._target_layers:
            target_layer_name = list(self._target_layers.keys())[0]
            print(f"Layer not found; falling back to '{target_layer_name}'")
        target_layer = self._target_layers[target_layer_name]

        try:
            cam          = EigenCAM(model=self._model.image_backbone, target_layers=[target_layer])
            targets      = [ClassifierOutputTarget(0)]
            grayscale    = cam(input_tensor=img_tensor, targets=cast(Any, targets))[0]
            grayscale    = self._normalize_cam(grayscale)
            cam_image    = self._overlay_heatmap(img_resized, grayscale)
            side_by_side = np.hstack((np.array(img_resized), cam_image))
            return cam_image, side_by_side
        except Exception:
            import traceback
            traceback.print_exc()
            return None, None

    @staticmethod
    def _normalize_cam(cam: np.ndarray) -> np.ndarray:
        lo, hi = cam.min(), cam.max()
        if hi != lo:
            return (cam - lo) / (hi - lo)
        return cam

    @staticmethod
    def _overlay_heatmap(img_resized, grayscale_cam: np.ndarray) -> np.ndarray:
        rgb_float = np.array(img_resized).astype(np.float32) / 255.0
        overlaid  = show_cam_on_image(
            rgb_float, grayscale_cam,
            use_rgb=True, colormap=cv2.COLORMAP_JET, image_weight=0.5,
        )
        return (overlaid * 255).astype(np.uint8)


    def build_interface(self) -> gr.Blocks:
        layer_choices = list(self._target_layers.keys())
        default_layer = layer_choices[1] if len(layer_choices) > 1 else layer_choices[0]

        with gr.Blocks(title="Melanoma Detection Explainability") as iface:
            gr.Markdown("# Melanoma Detection with Advanced Explainability")
            gr.Markdown(
                "Upload a skin lesion image to get a prediction along with a heatmap "
                "showing which regions influenced the model's decision."
            )

            with gr.Row():
                with gr.Column(scale=1):
                    image_input = gr.Image(type="pil", label="Upload Skin Lesion Image")

                    with gr.Group():
                        gr.Markdown("### Patient Metadata")
                        age_input  = gr.Number(label="Age", value=50)
                        sex_input  = gr.Dropdown(
                            label="Sex", choices=["male", "female", "unknown"], value="male"
                        )
                        site_input = gr.Dropdown(
                            label="Anatomical Site",
                            choices=["torso", "lower extremity", "upper extremity", "head/neck",
                                     "palms/soles", "oral/genital", "anterior torso",
                                     "posterior torso", "lateral torso", "unknown"],
                            value="torso",
                        )

                    with gr.Group():
                        gr.Markdown("### Visualization Options")
                        target_layer_input = gr.Dropdown(
                            label="Target Layer", choices=layer_choices, value=default_layer,
                            info="Earlier layers = more spatial detail; later = more semantic",
                        )

                    predict_btn = gr.Button("Get Prediction", variant="primary")

                with gr.Column(scale=2):
                    prediction_output  = gr.Textbox(label="Prediction")
                    gradcam_output     = gr.Image(label="EigenCAM Heatmap")
                    comparison_output  = gr.Image(label="Original vs. Heatmap")

            predict_btn.click(
                fn=self.predict_and_visualize,
                inputs=[image_input, age_input, sex_input, site_input, target_layer_input],
                outputs=[prediction_output, gradcam_output, comparison_output],
            )

        return iface

    def launch(self, **kwargs) -> None:
        self.build_interface().launch(**kwargs)
