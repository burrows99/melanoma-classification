import numpy as np
import pandas as pd

from config import Config


class MetadataPreprocessor:
    """Fits and applies preprocessing for tabular metadata columns.

    Numerical cols : median imputation → z-score standardisation (pandas/numpy)
    Categorical cols: 'Unknown' imputation → one-hot encoding aligned to train
                      categories so unseen val categories are silently zeroed out.

    Replaces sklearn SimpleImputer + StandardScaler + OneHotEncoder +
    ColumnTransformer + Pipeline with plain pandas/numpy — no sklearn needed here.
    """

    def __init__(self):
        self._num_medians: dict = {}
        self._num_means: dict = {}
        self._num_stds: dict = {}
        self._cat_categories: dict = {}  # col -> sorted list of known categories
        self._feature_cols: list = []
        self.num_output_features: int = 0

    def fit(self, train_df: pd.DataFrame) -> "MetadataPreprocessor":
        meta = Config.get_metadata_config()

        # Numerical: median for imputation, mean/std for scaling (fit on train only)
        for col in meta['numerical_cols']:
            s = pd.to_numeric(train_df[col], errors='coerce')
            self._num_medians[col] = s.median()
            filled = s.fillna(self._num_medians[col])
            self._num_means[col] = float(filled.mean())
            self._num_stds[col] = float(filled.std()) or 1.0  # avoid /0

        # Categorical: record known categories for consistent OHE column set
        for col in meta['categorical_cols']:
            self._cat_categories[col] = sorted(
                train_df[col].fillna('Unknown').astype(str).unique().tolist()
            )

        # Lock in the final column order using a single-row dry run
        self._feature_cols = self._build_features(train_df.iloc[:1]).columns.tolist()
        self.num_output_features = len(self._feature_cols)
        print(f"Metadata preprocessor fitted. Output features: {self.num_output_features}")
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        features = self._build_features(df)
        return features.reindex(columns=self._feature_cols, fill_value=0).values.astype(np.float32)

    def _build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        meta = Config.get_metadata_config()
        frames = []

        # Numerical: impute then standardise
        for col in meta['numerical_cols']:
            s = pd.to_numeric(df[col], errors='coerce').fillna(self._num_medians.get(col, 0))
            frames.append(
                ((s - self._num_means.get(col, 0)) / self._num_stds.get(col, 1))
                .rename(col).reset_index(drop=True)
            )

        # Categorical: impute → OHE → align to known categories (unknowns → 0)
        for col in meta['categorical_cols']:
            filled = df[col].fillna('Unknown').astype(str).reset_index(drop=True)
            dummies = pd.get_dummies(filled, prefix=col)
            known_cols = [f"{col}_{cat}" for cat in self._cat_categories.get(col, [])]
            for c in known_cols:
                if c not in dummies:
                    dummies[c] = 0
            frames.append(dummies[known_cols])

        return pd.concat(frames, axis=1)

