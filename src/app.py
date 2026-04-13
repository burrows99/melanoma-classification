import torch
import numpy as np
import cv2
from PIL import Image
import torchvision.transforms as T
import gradio as gr
import pandas as pd
import math
import torchvision.transforms.functional as TF

# Import existing model architecture
from model import get_model
from config import DEVICE

# GradCAM imports
from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# Configuration
CHECKPOINT_PATH = 'result/weights/gradcam.pth' 
IMAGE_SIZE = 256  

# Metadata defaults (for standalone predictions)
DEFAULT_METADATA = {
    'age_approx': 50.0,
    'sex': 'male',
    'anatom_site_general_challenge': 'torso'
}

# Create model with the same structure as during training
NUM_METADATA_FEATURES = 14  # This should match your original model
model = get_model(num_metadata_features=NUM_METADATA_FEATURES)
try:
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    print(f"✓ Model loaded from {CHECKPOINT_PATH}")
except Exception as e:
    print(f"Error loading model: {e}")
model.to(DEVICE).eval()

# Helper: find target layers for Grad-CAM at different depths
def get_target_layers(m):
    backbone = m.image_backbone
    layers = {}
    
    # Get a variety of layers at different depths
    # 1. Late layer (low resolution but high semantic information)
    if hasattr(backbone, 'conv_head'):
        layers['conv_head'] = backbone.conv_head
    
    # 2. Mid-level layers (more spatial resolution)
    if hasattr(backbone, 'blocks'):
        # Get third-to-last block
        if len(backbone.blocks) >= 3:
            block_idx = len(backbone.blocks) - 3
            if hasattr(backbone.blocks[block_idx], 'conv_pwl'):
                layers[f'blocks[-3].conv_pwl'] = backbone.blocks[block_idx].conv_pwl
        
        # Get middle block
        if len(backbone.blocks) >= 2:
            mid_idx = len(backbone.blocks) // 2
            if hasattr(backbone.blocks[mid_idx], 'conv_pwl'):
                layers[f'blocks[{mid_idx}].conv_pwl'] = backbone.blocks[mid_idx].conv_pwl
    
    # If nothing found, fallback to last Conv2d
    if not layers:
        for name, module in reversed(list(backbone.named_modules())):
            if isinstance(module, torch.nn.Conv2d):
                layers[name] = module
                break
    
    print(f"Found {len(layers)} target layers for Grad-CAM:")
    for name in layers.keys():
        print(f"  - {name}")
    
    return layers

# Get available target layers
target_layers = get_target_layers(model)

# --- TTA Transformations ---
# Define TTA transformations as lambdas for PIL Images
tta_transforms_pil = {
    'original': lambda img: img,
    'hflip': lambda img: img.transpose(Image.FLIP_LEFT_RIGHT),
    'vflip': lambda img: img.transpose(Image.FLIP_TOP_BOTTOM),
    'rot90': lambda img: img.rotate(90, expand=True), # expand=True to avoid cropping
    'rot180': lambda img: img.rotate(180, expand=True),
    'rot270': lambda img: img.rotate(270, expand=True),
}
# Note: Rotation might change aspect ratio if expand=True. If model expects square,


# Set up preprocessing
transform = T.Compose([
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Helper to prepare metadata input
def prepare_metadata(age=None, sex=None, site=None):
    # Use defaults for any missing values
    age = float(age) if age is not None else DEFAULT_METADATA['age_approx']
    sex = sex if sex else DEFAULT_METADATA['sex']
    site = site if site else DEFAULT_METADATA['anatom_site_general_challenge']
    
    # Create a metadata tensor (simplified version for demo 
    # This creates a basic one-hot encoding for categorical fields
    metadata = torch.zeros(NUM_METADATA_FEATURES, dtype=torch.float32)
    
    # Set age (assuming first position)
    metadata[0] = (age - 50.0) / 20.0  # Simple normalization around common age values
    
    # One-hot encode sex (assuming positions 1-3)
    if sex.lower() == 'male':
        metadata[1] = 1.0
    elif sex.lower() == 'female':
        metadata[2] = 1.0
    else:
        metadata[3] = 1.0  # unknown
    
    # One-hot encode anatomical site (assuming positions 4-13)
    site_mapping = {
        'torso': 4, 
        'lower extremity': 5, 
        'upper extremity': 6, 
        'head/neck': 7,
        'palms/soles': 8,
        'oral/genital': 9, 
        'anterior torso': 10, 
        'posterior torso': 11,
        'lateral torso': 12
    }
    
    if site.lower() in site_mapping:
        metadata[site_mapping[site.lower()]] = 1.0
    else:
        metadata[13] = 1.0  # unknown site
    
    return metadata.unsqueeze(0).to(DEVICE)  # Add batch dimension and move to device

# Main prediction + visualization function
def predict_and_visualize(img, age=None, sex='male', site='torso', 
                          target_layer_name='blocks[-3].conv_pwl'):
    if img is None:
        return "Please upload an image", None, None
    
    try:
        # Save original dimensions for later reference
        orig_width, orig_height = img.size
        print(f"Original image dimensions: {orig_width}x{orig_height}")
        
        # Preprocess image (original for visualization)
        img_rgb_original_pil = img.convert('RGB')
        
        # Prepare image for model (this will be used for CAM)
        img_for_cam_visualization = img_rgb_original_pil.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
        img_tensor_for_cam = transform(img_for_cam_visualization).unsqueeze(0).to(DEVICE)
        
        # Prepare metadata (remains constant for all TTA versions)
        metadata_tensor = prepare_metadata(age, sex, site)
        
        # --- Test Time Augmentation (TTA) ---
        all_probabilities = []
        print(f"Starting TTA with {len(tta_transforms_pil)} augmentations...")

        for aug_name, tta_pil_transform in tta_transforms_pil.items():
            try:
                # Apply TTA transformation to the original PIL image
                augmented_pil_img = tta_pil_transform(img_rgb_original_pil)
                
                # Apply standard preprocessing to the TTA version
                # Note: `transform` includes resize, to_tensor, normalize
                current_img_tensor = transform(augmented_pil_img).unsqueeze(0).to(DEVICE)
                
                with torch.no_grad():
                    logits = model(current_img_tensor, metadata_tensor)
                    probability = torch.sigmoid(logits).item()
                    all_probabilities.append(probability)
                    print(f"  TTA - {aug_name}: Probability = {probability:.4f}")
            except Exception as e_tta:
                print(f"  Error during TTA for {aug_name}: {e_tta}. Skipping this augmentation.")
                # Optionally, append a NaN or handle differently if one TTA version fails
                # For now, we just skip it.
        
        if not all_probabilities:
            return "Error: TTA failed for all augmentations.", None, None
            
        # Average the probabilities
        final_probability = np.mean(all_probabilities)
        print(f"TTA Final Averaged Probability: {final_probability:.4f} from {len(all_probabilities)} predictions.")
        
        # Simple output - just melanoma probability based on TTA
        prediction_text = f"Melanoma probability (TTA): {final_probability:.4f}"
        
        # Visualizations
        visualizations = {}
        
        # Select target layer
        if target_layer_name not in target_layers:
            # Default to first available layer
            target_layer = list(target_layers.values())[0]
            layer_name = list(target_layers.keys())[0]
            print(f"Selected layer '{target_layer_name}' not found. Using '{layer_name}' instead.")
        else:
            target_layer = target_layers[target_layer_name]
            print(f"Using '{target_layer_name}' for visualization")
        
        # 2) Generate visualization
        try:
            # Always target the melanoma class (class 0)
            target_category = 0
            targets = [ClassifierOutputTarget(target_category)]
            
            # Directly use EigenCAM
            print("Using EigenCAM for visualization.")
            cam_instance = EigenCAM(model=model.image_backbone, target_layers=[target_layer])
            
            # Get the CAM using the original image tensor
            grayscale_cam = cam_instance(input_tensor=img_tensor_for_cam, targets=targets) # USE IMG_TENSOR_FOR_CAM
            grayscale_cam = grayscale_cam[0, :]  # First image in batch
            
            # Print shape information for debugging
            print(f"Visualization shape (CAM output): {grayscale_cam.shape}")
            
            # Min-max normalize to ensure full [0-1] range
            if grayscale_cam.max() != grayscale_cam.min():
                grayscale_cam = (grayscale_cam - grayscale_cam.min()) / (grayscale_cam.max() - grayscale_cam.min())
            
            # Create perfectly aligned visualization
            # Use the CAM-prepared image (resized to IMAGE_SIZE x IMAGE_SIZE)
            img_array_for_viz = np.array(img_for_cam_visualization) 
            rgb_img_for_viz = img_array_for_viz.astype(np.float32) / 255.0
            
            # Apply the heatmap
            cam_image = show_cam_on_image(
                rgb_img_for_viz, # USE RGB_IMG_FOR_VIZ
                grayscale_cam,
                use_rgb=True,
                colormap=cv2.COLORMAP_JET,
                image_weight=0.5
            )
            
            # Convert to uint8 for display
            cam_image = (cam_image * 255).astype(np.uint8)
            
            # Generate side-by-side visualization
            # Use the CAM-prepared image for the "original" side in comparison
            orig_img_for_comparison = np.array(img_for_cam_visualization) 
            side_by_side = np.hstack((orig_img_for_comparison, cam_image))
            
            return prediction_text, cam_image, side_by_side
            
        except Exception as e:
            print(f"Error with visualization: {e}")
            import traceback
            traceback.print_exc()
            return prediction_text, None, None
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Error: {str(e)}", None, None

# Build Gradio interface
with gr.Blocks(title="Improved Melanoma Detection Explainability") as iface:
    gr.Markdown("# Melanoma Detection with Advanced Explainability")
    gr.Markdown("""
    Upload a skin lesion image to get a prediction along with visualizations that show
    which regions influenced the model's decision. These techniques help understand what
    the model is looking at and if it's finding genuine medical features or spurious correlations.
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            # Input controls
            image_input = gr.Image(type="pil", label="Upload Skin Lesion Image")
            
            with gr.Group():
                gr.Markdown("### Patient Metadata")
                age_input = gr.Number(label="Age", value=50)
                sex_input = gr.Dropdown(
                    label="Sex",
                    choices=["male", "female", "unknown"],
                    value="male"
                )
                site_input = gr.Dropdown(
                    label="Anatomical Site",
                    choices=["torso", "lower extremity", "upper extremity", "head/neck", 
                              "palms/soles", "oral/genital", "anterior torso", "posterior torso", 
                              "lateral torso", "unknown"],
                    value="torso"
                )
            
            with gr.Group():
                gr.Markdown("### Visualization Options")
                target_layer_input = gr.Dropdown(
                    label="Target Layer",
                    choices=list(target_layers.keys()),
                    value=list(target_layers.keys())[1] if len(target_layers) > 1 else list(target_layers.keys())[0],
                    info="Earlier layers (blocks) show more spatial detail, later layers show higher-level features"
                )
            
            predict_btn = gr.Button("Get Prediction", variant="primary")
            
        with gr.Column(scale=2):
            # Output displays
            prediction_output = gr.Textbox(label="Prediction")
            gradcam_output = gr.Image(label="Visualization")
            comparison_output = gr.Image(label="Original vs. Heatmap")
            
            with gr.Accordion("About the Visualizations", open=False):
                gr.Markdown("""
                ### Understanding the EigenCAM Visualization
                
                The heatmap highlights regions that the model's convolutional layers found most significant based on their principal components. This is an unsupervised method that shows general model attention.

                * **Red/Yellow regions**: Areas with stronger activations/influence on the model's feature extraction
                * **Blue regions**: Areas with less influence
                
                **Common issues in skin lesion classifiers that visualization can help identify:**
                1. **Border Artifacts**: Models may focus on the circular borders of dermoscopy images rather than the lesion itself.
                2. **Rulers/Color Charts**: These markers in the image can create unwanted correlations.
                3. **Hair/Bubbles**: Non-lesion features can sometimes be misleadingly predictive.
                
                **Target layers:**
                * **Earlier layers** (blocks in the middle): Higher resolution, more spatial detail.
                * **Later layers** (conv_head): More semantic information but lower resolution.
                """)
    
    # Connect the prediction function to the interface
    predict_btn.click(
        fn=predict_and_visualize,
        inputs=[image_input, age_input, sex_input, site_input, target_layer_input],
        outputs=[prediction_output, gradcam_output, comparison_output]
    )

if __name__ == "__main__":
    # Launch the interface
    iface.launch(share=False) 