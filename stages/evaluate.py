"""
Stage 4: Model Evaluation
Computes metrics, saves JSON report, and plots confusion matrix.
"""

import json
import logging
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)
from pathlib import Path

logger = logging.getLogger("pipeline.evaluate")


class ModelEvaluator:
    def __init__(self, config: dict):
        self.metric_names = config.get("metrics", ["accuracy", "f1"])
        self.save_report = config.get("save_report", True)
        self.report_path = config.get("report_path", "reports/evaluation.json")
        self.plot_cm = config.get("plot_confusion_matrix", True)

    def run(self, model, X_test, y_test) -> dict:
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

        metric_fns = {
            "accuracy":  lambda: accuracy_score(y_test, y_pred),
            "precision": lambda: precision_score(y_test, y_pred, average="weighted", zero_division=0),
            "recall":    lambda: recall_score(y_test, y_pred, average="weighted", zero_division=0),
            "f1":        lambda: f1_score(y_test, y_pred, average="weighted", zero_division=0),
            "roc_auc":   lambda: roc_auc_score(y_test, y_proba) if y_proba is not None else None,
        }

        metrics = {}
        for name in self.metric_names:
            if name in metric_fns:
                value = metric_fns[name]()
                if value is not None:
                    metrics[name] = round(float(value), 4)
                    logger.info(f"  {name}: {metrics[name]}")

        # Full classification report
        metrics["classification_report"] = classification_report(y_test, y_pred, output_dict=True)

        if self.save_report:
            Path(self.report_path).parent.mkdir(parents=True, exist_ok=True)
            with open(self.report_path, "w") as f:
                json.dump(metrics, f, indent=2)
            logger.info(f"Report saved to {self.report_path} ✓")

        if self.plot_cm:
            self._plot_confusion_matrix(y_test, y_pred)

        return metrics

    def _plot_confusion_matrix(self, y_test, y_pred):
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm, cmap="Blues")
        plt.colorbar(im, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix")

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")

        path = "reports/confusion_matrix.png"
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        logger.info(f"Confusion matrix saved to {path} ✓")
