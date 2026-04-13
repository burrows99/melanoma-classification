import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from config import Config
from file_io_manager import FileIOManager
from dataset.melanoma_dataset import MelanomaDataset
from dataset.metadata_preprocessor import MetadataPreprocessor
from dataset.transform import Transform


class MelanomaDataLoaders:
    """Builds and holds the train and validation DataLoaders.

    Usage:
        loaders = MelanomaDataLoaders()
        train_loader = loaders.get_train_loader()
        val_loader   = loaders.get_val_loader()
        n_features   = loaders.num_metadata_features
    """

    def __init__(self):
        df, image_paths = self._load_dataframe()
        train_df, val_df, train_img_paths, val_img_paths = self._split(df, image_paths)
        preprocessor = self._fit_preprocessor(train_df)
        self.num_metadata_features = preprocessor.num_output_features
        self._train_dataset = self._build_dataset(train_df, train_img_paths, preprocessor, train=True)
        self._val_dataset   = self._build_dataset(val_df,   val_img_paths,   preprocessor, train=False)

    def _load_dataframe(self):
        paths = Config.get_paths_config()
        try:
            df = pd.read_csv(paths['train_labels_path'])
        except FileNotFoundError:
            print(f"Error: Labels CSV not found at {paths['train_labels_path']}")
            raise
        image_paths = [
            FileIOManager.image_path(paths['train_data_dir'], img_name)
            for img_name in df['image_name']
        ]
        return df, image_paths

    def _split(self, df, image_paths):
        training = Config.get_training_config()
        indices = list(range(len(df)))
        train_indices, val_indices = train_test_split(
            indices,
            train_size=training['train_split'],
            random_state=training['random_seed'],
            stratify=df['target'].values,
        )
        train_df = df.iloc[train_indices]
        val_df   = df.iloc[val_indices]
        train_img_paths = [image_paths[i] for i in train_indices]
        val_img_paths   = [image_paths[i] for i in val_indices]
        print(f"Dataset split: {len(train_img_paths)} train, {len(val_img_paths)} validation samples.")
        return train_df, val_df, train_img_paths, val_img_paths

    def _fit_preprocessor(self, train_df):
        return MetadataPreprocessor().fit(train_df)

    def _build_dataset(self, df, img_paths, preprocessor, train: bool):
        return MelanomaDataset(
            img_paths, df['target'].values, preprocessor.transform(df),
            transform=Transform(train=train),
        )

    def get_train_loader(self) -> DataLoader:
        cfg = Config.get_training_config()
        return DataLoader(
            self._train_dataset,
            batch_size=cfg['batch_size'],
            shuffle=True,
            num_workers=cfg['num_workers'],
            pin_memory=cfg['device'] == 'cuda',
            drop_last=True,
        )

    def get_val_loader(self) -> DataLoader:
        cfg = Config.get_training_config()
        return DataLoader(
            self._val_dataset,
            batch_size=cfg['batch_size'],
            shuffle=False,
            num_workers=cfg['num_workers'],
            pin_memory=cfg['device'] == 'cuda',
        )

