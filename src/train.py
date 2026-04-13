import torch
from sklearn.metrics import accuracy_score, recall_score, f1_score
from tqdm import tqdm
from config import Config
from dataset import MelanomaDataLoaders
from model import MetadataMelanomaModel
from evaluate import Evaluator
from file_io_manager import FileIOManager


class Trainer:
    def __init__(self):
        self._device = Config.get_training_config()['device']

        loaders = MelanomaDataLoaders()
        self._train_loader = loaders.get_train_loader()
        self._val_loader   = loaders.get_val_loader()
        num_metadata_features = loaders.num_metadata_features

        self._model     = MetadataMelanomaModel.build(num_metadata_features=num_metadata_features)
        self._criterion = MetadataMelanomaModel.get_criterion()
        self._optimizer = MetadataMelanomaModel.get_optimizer(self._model)
        self._run_name  = self._build_run_name()
        self._io        = FileIOManager.for_run(Config.get_model_config()['architecture'])
        self._io.save_preprocessor(loaders.preprocessor)

    def _build_run_name(self) -> str:
        cfg = Config.get_training_config()
        base = (
            f"{Config.get_model_config()['architecture']}_Meta"
            f"_LR{cfg['learning_rate']}_BS{cfg['batch_size']}_Ep{cfg['num_epochs']}"
        )
        aug = Config.get_augmentation_config()
        if 'affine' in aug:
            suffix = "_AffAug"
        elif aug.get('random_erasing_prob', 0) > 0:
            suffix = "_EraseAug"
        else:
            suffix = "_BasicAug"
        return base + suffix

    def _train_epoch(self) -> tuple:
        self._model.train()
        total_loss = 0.0
        all_preds:  list[int] = []
        all_labels: list[int] = []

        for images, metadata, labels in tqdm(self._train_loader, desc="Training"):
            images   = images.to(self._device)
            metadata = metadata.to(self._device)
            labels   = labels.to(self._device)

            self._optimizer.zero_grad()
            outputs = self._model(images, metadata)
            loss    = self._criterion(outputs, labels.unsqueeze(1).float())
            loss.backward()
            self._optimizer.step()

            total_loss += loss.detach().item()
            preds = (torch.sigmoid(outputs.detach()) > 0.5).int()
            all_preds.extend(preds.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())

        return total_loss / len(self._train_loader), all_preds, all_labels

    def _save_checkpoint(self, epoch: int) -> str:
        path = self._io.save_checkpoint(self._model, self._run_name, epoch)
        return str(path)

    def _final_evaluation(self, best_model_path: str) -> None:
        print(f"Loading best model for final evaluation plots: {best_model_path}")
        final_model = MetadataMelanomaModel.build(num_metadata_features=self._model.num_metadata_features)
        self._io.load_checkpoint(final_model, best_model_path, map_location=self._device)
        final_model = final_model.to(self._device)
        evaluator = Evaluator(final_model, MetadataMelanomaModel.get_criterion(), io=self._io)
        _, preds, labels, probs = evaluator.evaluate(
            self._val_loader, use_tta=Config.get_evaluation_config()['tta_enabled']
        )
        evaluator.plot_roc_curve(labels, probs)
        evaluator.plot_confusion_matrix(labels, preds)

    def train(self) -> None:
        num_epochs     = Config.get_training_config()['num_epochs']
        best_val_f1    = 0.0
        best_path      = None
        best_epoch     = 0

        for epoch in range(num_epochs):
            current_lr = self._optimizer.param_groups[0]['lr']
            print(f"\nEpoch {epoch + 1}/{num_epochs}  |  LR: {current_lr}")

            train_loss, train_preds, train_labels = self._train_epoch()
            val_loss, val_preds, val_labels, _ = Evaluator(self._model, self._criterion).evaluate(
                self._val_loader, use_tta=Config.get_evaluation_config()['tta_enabled']
            )

            train_acc    = accuracy_score(train_labels, train_preds)
            train_recall = recall_score(train_labels, train_preds)
            train_f1     = f1_score(train_labels, train_preds)
            val_acc      = accuracy_score(val_labels, val_preds)
            val_recall   = recall_score(val_labels, val_preds)
            val_f1       = f1_score(val_labels, val_preds)

            print(f"Train: Loss={train_loss:.4f} | Acc={train_acc:.4f} | Recall={train_recall:.4f} | F1={train_f1:.4f}")
            print(f"Val:   Loss={val_loss:.4f} | Acc={val_acc:.4f} | Recall={val_recall:.4f} | F1={val_f1:.4f}")

            self._io.append_epoch_metrics({
                "epoch": epoch + 1,
                "learning_rate": current_lr,
                "train_loss": train_loss, "train_acc": train_acc,
                "train_recall": train_recall, "train_f1": train_f1,
                "val_loss": val_loss, "val_acc": val_acc,
                "val_recall": val_recall, "val_f1": val_f1,
            })

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_epoch  = epoch + 1
                best_path   = self._save_checkpoint(best_epoch)
                print(f"New best model saved: {best_path}  (Val F1: {best_val_f1:.4f})")

        if best_path:
            self._io.save_gradcam_checkpoint(self._model)
            print(f"Inference checkpoint saved: {self._io.gradcam_checkpoint_path()}")
            self._final_evaluation(best_path)
        else:
            print("Warning: No best model saved; skipping final plots.")
 