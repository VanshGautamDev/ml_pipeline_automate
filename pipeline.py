"""
Automated ML Pipeline Orchestrator
Run: python pipeline.py --config config.yaml
"""

import argparse
import yaml
import logging
from pathlib import Path

from stages.ingest import DataIngestor
from stages.preprocess import DataPreprocessor
from stages.train import ModelTrainer
from stages.evaluate import ModelEvaluator
from stages.deploy import ModelDeployer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("pipeline")


def run_pipeline(config: dict):
    logger.info("=== Starting ML Pipeline ===")

    # 1. Data Ingestion
    logger.info("Stage 1/4: Data Ingestion")
    ingestor = DataIngestor(config["data"])
    raw_data = ingestor.run()

    # 2. Preprocessing
    logger.info("Stage 2/4: Preprocessing")
    preprocessor = DataPreprocessor(config["preprocessing"])
    X_train, X_test, y_train, y_test = preprocessor.run(raw_data)

    # 3. Training & Tuning
    logger.info("Stage 3/4: Model Training & Tuning")
    trainer = ModelTrainer(config["training"])
    model = trainer.run(X_train, y_train)

    # 4. Evaluation
    logger.info("Stage 4/4: Evaluation")
    evaluator = ModelEvaluator(config["evaluation"])
    metrics = evaluator.run(model, X_test, y_test)
    logger.info(f"Evaluation metrics: {metrics}")

    # 5. Deploy (if threshold met)
    if metrics.get("accuracy", 0) >= config["deployment"]["min_accuracy"]:
        logger.info("Threshold met — deploying model")
        deployer = ModelDeployer(config["deployment"])
        deployer.run(model, metrics)
    else:
        logger.warning("Model did not meet deployment threshold. Skipping deploy.")

    logger.info("=== Pipeline Complete ===")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ML Pipeline")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    run_pipeline(config)
