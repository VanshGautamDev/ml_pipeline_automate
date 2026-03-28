"""
Stage 1: Data Ingestion
Supports CSV files, SQL databases, and S3 paths.
"""

import logging
import pandas as pd
from pathlib import Path

logger = logging.getLogger("pipeline.ingest")


class DataIngestor:
    def __init__(self, config: dict):
        self.source = config["source"]
        self.target_column = config["target_column"]
        self.test_size = config.get("test_size", 0.2)
        self.random_state = config.get("random_state", 42)

    def run(self) -> pd.DataFrame:
        logger.info(f"Loading data from: {self.source}")

        if self.source.endswith(".csv"):
            df = pd.read_csv(self.source)
        elif self.source.startswith(("postgresql://", "mysql://", "sqlite://")):
            df = self._load_from_db(self.source)
        elif self.source.startswith("s3://"):
            df = self._load_from_s3(self.source)
        else:
            raise ValueError(f"Unsupported data source: {self.source}")

        logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
        self._validate(df)
        return df

    def _load_from_db(self, url: str) -> pd.DataFrame:
        from sqlalchemy import create_engine
        engine = create_engine(url)
        # Customize your query here
        return pd.read_sql("SELECT * FROM dataset", engine)

    def _load_from_s3(self, path: str) -> pd.DataFrame:
        # Requires: pip install s3fs
        return pd.read_csv(path, storage_options={"anon": False})

    def _validate(self, df: pd.DataFrame):
        if self.target_column not in df.columns:
            raise ValueError(
                f"Target column '{self.target_column}' not found. "
                f"Available columns: {list(df.columns)}"
            )
        if df.empty:
            raise ValueError("Dataset is empty.")
        logger.info("Data validation passed ✓")
