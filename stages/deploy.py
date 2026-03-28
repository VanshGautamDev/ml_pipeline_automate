"""
Stage 5: Model Deployment
Saves model artifact + metadata. Optionally wraps in a FastAPI server.
"""

import json
import logging
import joblib
from datetime import datetime
from pathlib import Path

logger = logging.getLogger("pipeline.deploy")


class ModelDeployer:
    def __init__(self, config: dict):
        self.model_path = config.get("model_output_path", "models/model.pkl")
        self.save_metadata = config.get("save_metadata", True)

    def run(self, model, metrics: dict):
        # Model already saved by trainer; just log and write metadata
        metadata = {
            "deployed_at": datetime.utcnow().isoformat(),
            "model_type": type(model).__name__,
            "model_path": self.model_path,
            "metrics": {k: v for k, v in metrics.items() if k != "classification_report"},
        }

        if self.save_metadata:
            meta_path = Path(self.model_path).parent / "metadata.json"
            with open(meta_path, "w") as f:
                json.dump(metadata, f, indent=2)
            logger.info(f"Metadata saved to {meta_path} ✓")

        logger.info("Model deployed successfully ✓")
        logger.info("To serve the model, run:  uvicorn serve:app --reload")
        return metadata


# ── Optional FastAPI inference server ──────────────────────────────────────────
# Run standalone:  uvicorn stages.deploy:app --reload
# Then POST to:    http://localhost:8000/predict

try:
    from fastapi import FastAPI
    from pydantic import BaseModel
    import pandas as pd

    app = FastAPI(title="ML Pipeline — Inference Server")

    _model = None
    _preprocessor = None

    def _load_artifacts():
        global _model, _preprocessor
        _model = joblib.load("models/model.pkl")
        _preprocessor = joblib.load("models/preprocessor.pkl")

    class PredictRequest(BaseModel):
        features: dict  # {"col1": value, "col2": value, ...}

    @app.on_event("startup")
    def startup():
        _load_artifacts()

    @app.post("/predict")
    def predict(request: PredictRequest):
        df = pd.DataFrame([request.features])
        X = _preprocessor.transform(df)
        prediction = _model.predict(X)[0]
        proba = _model.predict_proba(X)[0].tolist() if hasattr(_model, "predict_proba") else None
        return {"prediction": int(prediction), "probabilities": proba}

    @app.get("/health")
    def health():
        return {"status": "ok", "model": type(_model).__name__}

except ImportError:
    # FastAPI not installed — server not available
    pass
