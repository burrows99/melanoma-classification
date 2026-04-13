import torch
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, f1_score # Keep only necessary metrics here
from tqdm import tqdm
import os
import wandb
from torch.utils.data import DataLoader
from config import * # Imports DEVICE, TTA_ENABLED_EVAL, etc.
from dataset import get_data_loaders # get_image_transforms is used in evaluate.py now
from model import get_model, get_criterion, get_optimizer
# Import functions from the new evaluate.py
from evaluate import evaluate, plot_roc_curve, plot_confusion_matrix

# --- TTA Transformations and val_transforms are now in evaluate.py ---

# --- Training & Evaluation Utilities ---
def train_epoch(model, train_loader, criterion, optimizer):
    model.train() # Set model to training mode
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    # Unpack image, metadata, and label
    for images, metadata, labels in tqdm(train_loader, desc="Training"):
        images = images.to(DEVICE)
        metadata = metadata.to(DEVICE) # Move metadata to device
        labels = labels.to(DEVICE)
        
        optimizer.zero_grad()
        # Pass image and metadata to model
        outputs = model(images, metadata)
        loss = criterion(outputs, labels.unsqueeze(1).float()) 
        loss.backward()
        optimizer.step()
        
        total_loss += loss.detach().item()
        preds = (torch.sigmoid(outputs.detach()) > 0.5).int()
        all_preds.extend(preds.cpu().numpy().flatten()) # Flatten in case of shape [N, 1]
        all_labels.extend(labels.cpu().numpy().flatten())
    
    return total_loss / len(train_loader), all_preds, all_labels

# --- Main Training Function ---
def train_model():
    # --- Setup ---
    
    run_name_suffix = ""
    # Logic for augmentation part of name (can be kept or simplified)
    if 'affine' in AUGMENTATION: run_name_suffix += "_AffAug"
    elif 'random_erasing_prob' in AUGMENTATION and AUGMENTATION['random_erasing_prob'] > 0: run_name_suffix += "_EraseAug"
    else: run_name_suffix += "_BasicAug"
    
    # Base run name without scheduler indication initially
    base_run_name = f"{MODEL_ARCHITECTURE}_Meta_LR{LEARNING_RATE}_BS{BATCH_SIZE}_Ep{NUM_EPOCHS}"
    run_name = base_run_name + run_name_suffix

    # --- Initialize Scheduler (IF USED) ---
    scheduler = None # Default to no scheduler
    

    # --- WandB Initialization ---
    run = wandb.init(
        project="melanoma-classification",
        name=run_name, 
        config={
            "architecture": MODEL_ARCHITECTURE + " + Metadata MLP", 
            "image_size": IMAGE_SIZE,
            "epochs": NUM_EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "optimizer": "Adam",
            "loss_function": LOSS_FUNCTION_TYPE,
            "tta_enabled_eval": TTA_ENABLED_EVAL # Log if TTA is used in eval
            
        },
        settings=wandb.Settings(start_method="thread")
    )
        
    train_loader, val_loader, num_metadata_features = get_data_loaders()
    if "num_metadata_features" not in wandb.config: # Only update if not already set (e.g. by USE_METADATA logic)
        wandb.config.update({"num_metadata_features": num_metadata_features})

    model = get_model(num_metadata_features=num_metadata_features)
    model = model.to(DEVICE)
    
    criterion = get_criterion()
    optimizer = get_optimizer(model, learning_rate=LEARNING_RATE) # Use LEARNING_RATE from config as initial

    # --- Training Loop ---
    best_val_f1 = 0.0 
    best_model_local_path = None 
    overall_best_epoch = 0
    
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        # Log current LR - if scheduler is None, this will just be the fixed LEARNING_RATE
        current_lr = optimizer.param_groups[0]['lr'] 
        print(f"Current Learning Rate: {current_lr}")

        train_loss, train_preds, train_labels = train_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_preds, val_labels, val_probs = evaluate(model, val_loader, criterion, use_tta=TTA_ENABLED_EVAL)
        
        # -- Calculate Metrics --
        train_acc = accuracy_score(train_labels, train_preds)
        train_recall = recall_score(train_labels, train_preds)
        train_f1 = f1_score(train_labels, train_preds)
        val_acc = accuracy_score(val_labels, val_preds)
        val_recall = recall_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds)
        
        # -- Logging to W&B and Console --
        log_dict = {
            "train/loss": train_loss,
            "train/accuracy": train_acc,
            "train/recall": train_recall,
            "train/f1": train_f1,
            "val/loss": val_loss,
            "val/accuracy": val_acc,
            "val/recall": val_recall,
            "val/f1": val_f1,
            "learning_rate": current_lr,
            "epoch": epoch + 1  # Explicitly log the epoch number
        }
        wandb.log(log_dict)
        print(f"Train Metrics: Loss={train_loss:.4f} | Acc={train_acc:.4f} | Recall={train_recall:.4f} | F1={train_f1:.4f}")
        print(f"Val Metrics:   Loss={val_loss:.4f} | Acc={val_acc:.4f} | Recall={val_recall:.4f} | F1={val_f1:.4f}")
        
        # -- Learning Rate Scheduler Step (IF USED) ---
        # if scheduler is not None:
        #     scheduler.step(val_loss)

        # -- Save Best Model (based on validation F1) --
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            overall_best_epoch = epoch + 1
            model_name_for_saving = f"result/weights/{run_name}_best_ep{overall_best_epoch}.pth" 
            os.makedirs('result/weights', exist_ok=True)
            torch.save(model.state_dict(), model_name_for_saving)
            print(f"New best model saved: {model_name_for_saving} (Val F1: {best_val_f1:.4f} at epoch {overall_best_epoch})")
            best_model_local_path = model_name_for_saving
            wandb.run.summary["best_f1"] = best_val_f1
            wandb.run.summary["best_epoch"] = overall_best_epoch
            wandb.run.summary["best_val_loss"] = val_loss
            wandb.run.summary["best_val_accuracy"] = val_acc
            wandb.run.summary["best_val_recall"] = val_recall
            wandb.run.summary["best_train_loss"] = train_loss
            wandb.run.summary["best_train_accuracy"] = train_acc
            wandb.run.summary["best_train_f1"] = train_f1
            wandb.run.summary["best_train_recall"] = train_recall

    # --- Post-Training Operations ---
    if best_model_local_path:
        print(f"Logging final best model to W&B artifact: {best_model_local_path} from epoch {overall_best_epoch}")
        artifact = wandb.Artifact(
            name=f"model-{run_name}", 
            type="model",
            description=f"Best model from run {run_name} with F1={best_val_f1:.4f} at epoch {overall_best_epoch}",
            metadata=wandb.config.as_dict()
        )
        artifact.add_file(best_model_local_path)
        run.log_artifact(artifact)
        
        print(f"Loading overall best model for final evaluation plots: {best_model_local_path}")
        final_model = get_model(num_metadata_features=num_metadata_features) 
        final_model.load_state_dict(torch.load(best_model_local_path))
        final_model = final_model.to(DEVICE)
        final_criterion = get_criterion()
        _, final_val_preds_for_cm, final_val_labels_for_plot, final_val_probs_for_plot = evaluate(
            final_model, val_loader, final_criterion, use_tta=TTA_ENABLED_EVAL
        ) 
        plot_roc_curve(final_val_labels_for_plot, final_val_probs_for_plot)
        plot_confusion_matrix(final_val_labels_for_plot, final_val_preds_for_cm)
        wandb.log({
            "roc_curve": wandb.Image('roc_curve.png'),
            "confusion_matrix": wandb.Image('confusion_matrix.png')
        })
    else:
        print("Warning: No best model was saved/identified; skipping artifact logging and final plots.")

    wandb.finish()

if __name__ == "__main__":
    # TTA_ENABLED_EVAL is imported from config.py and its status will be in wandb logs.
    train_model() 