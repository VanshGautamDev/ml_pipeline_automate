# 🤖 AutoML Pipeline

An end-to-end automated machine learning pipeline built with Python and scikit-learn. Drop in your dataset, run one command, and get a trained, evaluated, and deployed model — automatically.

---

## ✨ Features

- **Auto model selection** — trains 13 models and picks the best one
- **Stratified K-Fold CV** — fair evaluation across all folds
- **Overfitting / underfitting detection** — flags and penalises unhealthy models
- **Composite health scoring** — selects by generalisation, not just raw accuracy
- **Smart hyperparameter tuning** — GridSearchCV per model
- **Full comparison report** — JSON + charts for every model trained
- **FastAPI inference server** — serve predictions via REST API
- **One config file** — control everything from `config.yaml`

---

## 📁 Project Structure

```
ml_pipeline/
├── pipeline.py              # Orchestrator — run this
├── config.yaml              # All settings in one place
├── requirements.txt         # Dependencies
│
├── stages/
│   ├── ingest.py            # Stage 1 — Load data (CSV / DB / S3)
│   ├── preprocess.py        # Stage 2 — Clean, scale, encode
│   ├── train.py             # Stage 3 — Train all models, pick best
│   ├── evaluate.py          # Stage 4 — Metrics, report, confusion matrix
│   └── deploy.py            # Stage 5 — Save artifacts + FastAPI server
│
├── data/                    # Put your raw dataset here
├── models/                  # Saved model.pkl + preprocessor.pkl
└── reports/                 # Evaluation JSON + charts
```

---

## 🚀 Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Add your dataset

```bash
cp your_dataset.csv data/raw.csv
```

### 3. Configure

Open `config.yaml` and set at minimum:

```yaml
data:
  source: "data/raw.csv"
  target_column: "your_label_column"
```

### 4. Run the pipeline

```bash
python pipeline.py --config config.yaml
```

That's it. The pipeline will ingest, preprocess, train all models, evaluate, and deploy the best one.

---

## 🧠 How Model Selection Works

The pipeline trains **13 models** simultaneously:

| Category | Models |
|---|---|
| Linear | Logistic Regression, Ridge Classifier, SGD Classifier |
| Tree-based | Decision Tree, Random Forest, Extra Trees, Gradient Boosting, AdaBoost, Bagging |
| Other | SVM, KNN, Naive Bayes, LDA |

Each model goes through:

1. **GridSearchCV** — finds the best hyperparameters
2. **Stratified K-Fold CV** — evaluates on each fold, records train + test scores
3. **Health diagnosis** — checks for 3 problems:

| Problem | Condition | What it means |
|---|---|---|
| Overfitting | `train_score - cv_score > 0.08` | Model memorised training data |
| Underfitting | `cv_score < 0.60` | Model too simple to learn |
| High Variance | `cv_std > 0.05` | Unstable — results vary wildly across folds |

4. **Health score** — composite ranking formula:

```
health_score = cv_accuracy
             − (overfit_gap  × 0.5)   # penalise memorisation
             − (cv_std       × 0.5)   # penalise instability
             − (num_issues   × 0.02)  # deduction per problem found
```

The model with the **highest health score** wins — not just the highest raw accuracy. Underfitting models are excluded from the final pool entirely.

---

## 📊 Reports Generated

After every run, the following files are saved to `reports/`:

| File | Description |
|---|---|
| `model_comparison.json` | Full stats (cv_mean, cv_std, gap, issues, health_score) for every model |
| `model_comparison.png` | Bar charts — CV accuracy + health score (winner highlighted) |
| `evaluation.json` | Final metrics on the held-out test set |
| `confusion_matrix.png` | Confusion matrix of the winning model |

---

## ⚙️ Configuration Reference

```yaml
data:
  source: "data/raw.csv"       # Path to CSV, a DB URL (postgresql://...), or S3 path (s3://...)
  target_column: "label"       # Name of the column you want to predict
  test_size: 0.2               # 20% held out for final evaluation
  random_state: 42

preprocessing:
  drop_duplicates: true
  handle_missing: "median"     # median | mean | mode | drop
  scale_features: true         # StandardScaler on numeric columns
  encode_categoricals: true    # OneHotEncoder on categorical columns

training:
  models: "all"                # "all"  OR a subset: [random_forest, svm, knn]
  hyperparameter_tuning: true  # Run GridSearchCV per model
  cv_folds: 5                  # Number of Stratified K-Fold splits
  scoring: "accuracy"          # Metric used during CV

  # Diagnostic thresholds
  overfit_threshold: 0.08      # train-cv gap above this → overfitting
  underfit_threshold: 0.60     # cv score below this     → underfitting
  variance_threshold: 0.05     # cv std above this       → high variance

  model_output_path: "models/model.pkl"

evaluation:
  metrics: [accuracy, precision, recall, f1, roc_auc]
  save_report: true
  report_path: "reports/evaluation.json"
  plot_confusion_matrix: true

deployment:
  min_accuracy: 0.80           # Only deploy if accuracy >= this threshold
  model_output_path: "models/model.pkl"
  save_metadata: true
```

---

## 🌐 Serving Predictions (FastAPI)

Install the server dependencies:

```bash
pip install fastapi uvicorn
```

Start the server:

```bash
uvicorn stages.deploy:app --reload
```

Send a prediction request:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": {"col1": 1.5, "col2": "category_A", "col3": 42}}'
```

Response:

```json
{
  "prediction": 1,
  "probabilities": [0.12, 0.88]
}
```

Check server health:

```bash
curl http://localhost:8000/health
```

---

## 🔌 Supported Data Sources

| Source | Example |
|---|---|
| CSV file | `"data/raw.csv"` |
| PostgreSQL | `"postgresql://user:pass@host/dbname"` |
| MySQL | `"mysql://user:pass@host/dbname"` |
| SQLite | `"sqlite:///path/to/db.sqlite"` |
| AWS S3 | `"s3://bucket-name/path/to/file.csv"` |

For database sources, the ingestor runs `SELECT * FROM dataset` — edit `stages/ingest.py` to customise the query.

---

## 🛠️ Extending the Pipeline

### Add a new model

In `stages/train.py`, add to `MODEL_REGISTRY`:

```python
from sklearn.neural_network import MLPClassifier

MODEL_REGISTRY["mlp"] = MLPClassifier(random_state=42, max_iter=500)
```

And optionally add its param grid to `PARAM_GRIDS`:

```python
PARAM_GRIDS["mlp"] = {
    "hidden_layer_sizes": [(64,), (128,), (64, 64)],
    "alpha": [0.0001, 0.001],
}
```

### Add a new metric

In `stages/evaluate.py`, add to `metric_fns`:

```python
from sklearn.metrics import matthews_corrcoef

metric_fns["mcc"] = lambda: matthews_corrcoef(y_test, y_pred)
```

Then add `"mcc"` to `metrics` in `config.yaml`.

### Use a subset of models (faster runs)

```yaml
training:
  models: [random_forest, gradient_boosting, svm]
```

---

## 📦 Installation for Development

```bash
git clone https://github.com/yourname/automl-pipeline
cd automl-pipeline
pip install -e .
```

---

## 🧪 Running Tests

```bash
pip install pytest
pytest tests/
```

---

## 📋 Requirements

- Python 3.9+
- scikit-learn >= 1.3
- pandas >= 2.0
- numpy >= 1.24
- matplotlib >= 3.7
- pyyaml >= 6.0
- joblib >= 1.3

Optional:
- `fastapi` + `uvicorn` — for the inference server
- `sqlalchemy` — for database ingestion
- `s3fs` — for S3 ingestion

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

## 🤝 Contributing

1. Fork the repo
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Commit your changes: `git commit -m "add my feature"`
4. Push and open a Pull Request

Please make sure new code includes tests and the pipeline runs end-to-end before submitting.
