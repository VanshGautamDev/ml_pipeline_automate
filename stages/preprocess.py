"""
Stage 2: Data Preprocessing
Handles missing values, feature scaling, and categorical encoding.
"""

import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib

logger = logging.getLogger("pipeline.preprocess")


class DataPreprocessor:
    def __init__(self, config: dict):
        self.drop_duplicates = config.get("drop_duplicates", True)
        self.handle_missing = config.get("handle_missing", "median")
        self.scale_features = config.get("scale_features", True)
        self.encode_categoricals = config.get("encode_categoricals", True)
        self.preprocessor = None  # sklearn ColumnTransformer (saved for inference)

    def run(self, df: pd.DataFrame, target_column: str = "label"):
        # Pull target column name from the dataframe's last stage config
        # (passed through by the orchestrator — adjust if needed)
        if target_column not in df.columns:
            target_column = df.columns[-1]  # fallback: last column
            logger.warning(f"Defaulting to last column as target: '{target_column}'")

        # Drop duplicates
        if self.drop_duplicates:
            before = len(df)
            df = df.drop_duplicates()
            logger.info(f"Dropped {before - len(df)} duplicate rows")

        X = df.drop(columns=[target_column])
        y = df[target_column]

        # Identify column types
        numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
        categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

        logger.info(f"Numeric features: {len(numeric_cols)}, Categorical: {len(categorical_cols)}")

        # Build pipelines
        numeric_steps = [("imputer", SimpleImputer(strategy=self.handle_missing))]
        if self.scale_features:
            numeric_steps.append(("scaler", StandardScaler()))

        categorical_steps = [
            ("imputer", SimpleImputer(strategy="most_frequent")),
        ]
        if self.encode_categoricals:
            categorical_steps.append(
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
            )

        transformers = []
        if numeric_cols:
            transformers.append(("num", Pipeline(numeric_steps), numeric_cols))
        if categorical_cols:
            transformers.append(("cat", Pipeline(categorical_steps), categorical_cols))

        self.preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

        # Split then fit/transform (avoid data leakage)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y if y.nunique() < 20 else None
        )

        X_train = self.preprocessor.fit_transform(X_train)
        X_test = self.preprocessor.transform(X_test)

        # Save preprocessor for inference
        joblib.dump(self.preprocessor, "models/preprocessor.pkl")
        logger.info(f"Train shape: {X_train.shape} | Test shape: {X_test.shape}")

        return X_train, X_test, y_train, y_test
