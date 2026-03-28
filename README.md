# Automated ML Pipeline

End-to-end sklearn pipeline: ingest → preprocess → train → evaluate → deploy.

## Project Structure

```
ml_pipeline/
├── pipeline.py          # Orchestrator — run this
├── config.yaml          # All settings in one place
├── requirements.txt
├── stages/
│   ├── ingest.py        # CSV / DB / S3 loading
│   ├── preprocess.py    # Cleaning, scaling, encoding
│   ├── train.py         # Training + GridSearchCV tuning
│   ├── evaluate.py      # Metrics, report, confusion matrix
│   └── deploy.py        # Save artifacts + FastAPI server
├── data/                # Put your raw CSV here
├── models/              # Saved model.pkl + preprocessor.pkl
└── reports/             # evaluation.json + confusion_matrix.png
```

## Quickstart

```bash
pip install -r requirements.txt

# Add your data
cp your_dataset.csv data/raw.csv

# Edit config.yaml (set target_column, model_type, etc.)

# Run the full pipeline
python pipeline.py --config config.yaml
```

## Serve Predictions (Optional)

```bash
pip install fastapi uvicorn
uvicorn stages.deploy:app --reload
```

Then POST to `http://localhost:8000/predict`:
```json
{ "features": { "col1": 1.5, "col2": "category_A" } }
```

## Configuration

All behavior is controlled via `config.yaml`:

| Section | Key | Description |
|---|---|---|
| `data` | `source` | Path to CSV, DB URL, or S3 path |
| `data` | `target_column` | Name of the label/target column |
| `preprocessing` | `handle_missing` | `median`, `mean`, `mode`, or `drop` |
| `training` | `model_type` | `random_forest`, `logistic_regression`, `gradient_boosting` |
| `training` | `hyperparameter_tuning` | `true` / `false` |
| `deployment` | `min_accuracy` | Minimum accuracy to trigger deployment |

## Extending the Pipeline

- **New model**: Add to `MODEL_REGISTRY` in `stages/train.py`
- **New data source**: Add a branch in `stages/ingest.py`
- **New metric**: Add to `metric_fns` dict in `stages/evaluate.py`
