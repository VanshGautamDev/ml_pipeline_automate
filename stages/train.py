"""
Stage 3: Model Training — Auto Model Selection
- Trains ALL models using Stratified K-Fold CV + GridSearchCV tuning
- Detects overfitting, underfitting, high variance
- Selects best model by a composite health score (not just raw accuracy)
- Saves full comparison report to reports/model_comparison.json
"""

import json
import logging
import numpy as np
import matplotlib.pyplot as plt
import joblib
from pathlib import Path

from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    ExtraTreesClassifier, AdaBoostClassifier, BaggingClassifier
)
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import (
    StratifiedKFold, GridSearchCV, cross_validate
)

logger = logging.getLogger("pipeline.train")

# ── Model Registry ─────────────────────────────────────────────────────────────

MODEL_REGISTRY = {
    "logistic_regression":  LogisticRegression(max_iter=1000, random_state=42),
    "ridge_classifier":     RidgeClassifier(),
    "sgd_classifier":       SGDClassifier(random_state=42, max_iter=1000),
    "decision_tree":        DecisionTreeClassifier(random_state=42),
    "random_forest":        RandomForestClassifier(random_state=42, n_jobs=-1),
    "extra_trees":          ExtraTreesClassifier(random_state=42, n_jobs=-1),
    "gradient_boosting":    GradientBoostingClassifier(random_state=42),
    "adaboost":             AdaBoostClassifier(random_state=42, algorithm="SAMME"),
    "bagging":              BaggingClassifier(random_state=42, n_jobs=-1),
    "svm":                  SVC(probability=True, random_state=42),
    "knn":                  KNeighborsClassifier(n_jobs=-1),
    "naive_bayes":          GaussianNB(),
    "lda":                  LinearDiscriminantAnalysis(),
}

PARAM_GRIDS = {
    "logistic_regression":  {"C": [0.01, 0.1, 1, 10], "solver": ["lbfgs", "liblinear"]},
    "ridge_classifier":     {"alpha": [0.1, 1.0, 10.0]},
    "sgd_classifier":       {"alpha": [0.0001, 0.001, 0.01], "loss": ["hinge", "log_loss"]},
    "decision_tree":        {"max_depth": [3, 5, 10, None], "min_samples_split": [2, 5, 10]},
    "random_forest":        {"n_estimators": [100, 200], "max_depth": [5, 10, None], "min_samples_split": [2, 5]},
    "extra_trees":          {"n_estimators": [100, 200], "max_depth": [5, 10, None]},
    "gradient_boosting":    {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1], "max_depth": [3, 5]},
    "adaboost":             {"n_estimators": [50, 100, 200], "learning_rate": [0.5, 1.0]},
    "bagging":              {"n_estimators": [10, 50], "max_samples": [0.8, 1.0]},
    "svm":                  {"C": [0.1, 1, 10], "kernel": ["rbf", "linear"]},
    "knn":                  {"n_neighbors": [3, 5, 7, 11], "weights": ["uniform", "distance"]},
    "naive_bayes":          {},
    "lda":                  {"solver": ["svd", "lsqr"]},
}

# ── Diagnostic Thresholds ──────────────────────────────────────────────────────

OVERFIT_THRESHOLD  = 0.08   # train - cv > this  → overfitting
UNDERFIT_THRESHOLD = 0.60   # cv < this           → underfitting
VARIANCE_THRESHOLD = 0.05   # cv_std > this       → high variance


class ModelTrainer:
    def __init__(self, config: dict):
        self.tune        = config.get("hyperparameter_tuning", True)
        self.cv_folds    = config.get("cv_folds", 5)
        self.scoring     = config.get("scoring", "accuracy")
        self.models_cfg  = config.get("models", "all")
        self.output_path = config.get("model_output_path", "models/model.pkl")

    # ── Main entry point ──────────────────────────────────────────────────────

    def run(self, X_train, y_train):
        Path("models").mkdir(exist_ok=True)
        Path("reports").mkdir(exist_ok=True)

        candidates = self._select_candidates()
        logger.info(f"Training {len(candidates)} models with {self.cv_folds}-fold Stratified K-Fold CV\n")

        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        results = {}

        for name, base_model in candidates.items():
            logger.info(f"  ▶  {name}")
            result = self._train_one(name, base_model, X_train, y_train, cv)
            results[name] = result
            self._log_result(name, result)

        self._save_report(results)
        self._plot_comparison(results)

        best_name, best_model = self._select_best(results)
        logger.info(
            f"\n✅  Best model selected: [{best_name}]"
            f"  cv={results[best_name]['cv_mean']:.4f}"
            f"  health={results[best_name]['health_score']:.4f}"
        )

        joblib.dump(best_model, self.output_path)
        logger.info(f"Saved → {self.output_path} ✓")
        return best_model

    # ── Train one model ───────────────────────────────────────────────────────

    def _train_one(self, name, base_model, X_train, y_train, cv) -> dict:
        param_grid = PARAM_GRIDS.get(name, {})
        model = base_model
        best_params = {}

        # Hyperparameter tuning via GridSearchCV
        if self.tune and param_grid:
            search = GridSearchCV(
                base_model, param_grid,
                cv=cv, scoring=self.scoring,
                n_jobs=-1, refit=True
            )
            try:
                search.fit(X_train, y_train)
                model = search.best_estimator_
                best_params = search.best_params_
            except Exception as e:
                logger.warning(f"    GridSearch failed for {name}: {e} — using defaults")
                model.fit(X_train, y_train)
        else:
            model.fit(X_train, y_train)

        # K-Fold cross-validation (both train + test scores to detect overfit)
        cv_result = cross_validate(
            model, X_train, y_train,
            cv=cv,
            scoring=self.scoring,
            return_train_score=True,
            n_jobs=-1
        )

        train_mean = float(np.mean(cv_result["train_score"]))
        cv_mean    = float(np.mean(cv_result["test_score"]))
        cv_std     = float(np.std(cv_result["test_score"]))
        gap        = train_mean - cv_mean

        issues = self._diagnose(train_mean, cv_mean, cv_std, gap)

        return {
            "model":        model,
            "best_params":  best_params,
            "train_mean":   round(train_mean, 4),
            "cv_mean":      round(cv_mean, 4),
            "cv_std":       round(cv_std, 4),
            "gap":          round(gap, 4),
            "issues":       issues,
            "health_score": self._health_score(cv_mean, cv_std, gap, issues),
            "fold_scores":  [round(s, 4) for s in cv_result["test_score"]],
        }

    # ── Diagnose model health ─────────────────────────────────────────────────

    def _diagnose(self, train_mean, cv_mean, cv_std, gap) -> list:
        issues = []

        if gap > OVERFIT_THRESHOLD:
            issues.append(
                f"OVERFITTING — train={train_mean:.3f} vs cv={cv_mean:.3f} "
                f"(gap={gap:.3f}, threshold={OVERFIT_THRESHOLD})"
            )
        if cv_mean < UNDERFIT_THRESHOLD:
            issues.append(
                f"UNDERFITTING — cv={cv_mean:.3f} < threshold={UNDERFIT_THRESHOLD}"
            )
        if cv_std > VARIANCE_THRESHOLD:
            issues.append(
                f"HIGH VARIANCE — cv_std={cv_std:.3f} > threshold={VARIANCE_THRESHOLD} "
                "(unstable across folds)"
            )

        return issues

    # ── Composite health score ─────────────────────────────────────────────────
    # Penalises overfitting gap, high variance, and number of issues.
    # Models with the highest health score get selected — not just raw accuracy.

    def _health_score(self, cv_mean, cv_std, gap, issues) -> float:
        score = cv_mean
        score -= min(gap, 0.20) * 0.5      # penalise overfit gap (up to -0.10)
        score -= min(cv_std, 0.10) * 0.5   # penalise high variance (up to -0.05)
        score -= len(issues) * 0.02        # small deduction per issue
        return round(max(0.0, score), 4)

    # ── Select best model ─────────────────────────────────────────────────────

    def _select_best(self, results: dict):
        # Exclude clearly underfitting models from final pool
        healthy = {
            n: r for n, r in results.items()
            if not any("UNDERFITTING" in i for i in r["issues"])
        }
        pool = healthy if healthy else results  # fallback if all underfit

        best_name = max(pool, key=lambda n: pool[n]["health_score"])
        return best_name, pool[best_name]["model"]

    # ── Candidate filtering ───────────────────────────────────────────────────

    def _select_candidates(self) -> dict:
        if self.models_cfg == "all":
            return MODEL_REGISTRY
        return {k: v for k, v in MODEL_REGISTRY.items() if k in self.models_cfg}

    # ── Logging ───────────────────────────────────────────────────────────────

    def _log_result(self, name, r):
        if r["issues"]:
            status = "⚠️  " + " | ".join(r["issues"])
        else:
            status = "✓  Healthy"
        logger.info(
            f"    {name:25s} cv={r['cv_mean']:.4f}±{r['cv_std']:.4f}"
            f"  train={r['train_mean']:.4f}  gap={r['gap']:.4f}"
            f"  health={r['health_score']:.4f}  {status}"
        )

    # ── Persist comparison report ─────────────────────────────────────────────

    def _save_report(self, results: dict):
        report = {
            name: {k: v for k, v in r.items() if k != "model"}
            for name, r in results.items()
        }
        with open("reports/model_comparison.json", "w") as f:
            json.dump(report, f, indent=2)
        logger.info("\nModel comparison report → reports/model_comparison.json ✓")

    # ── Plot bar charts ───────────────────────────────────────────────────────

    def _plot_comparison(self, results: dict):
        names  = list(results.keys())
        means  = [r["cv_mean"]      for r in results.values()]
        stds   = [r["cv_std"]       for r in results.values()]
        health = [r["health_score"] for r in results.values()]
        colors = ["#e74c3c" if r["issues"] else "#2ecc71" for r in results.values()]

        x = np.arange(len(names))
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))

        # ── Top: CV Accuracy with error bars ──
        axes[0].bar(x, means, yerr=stds, capsize=4, color=colors, alpha=0.85)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(names, rotation=35, ha="right", fontsize=9)
        axes[0].set_ylabel("CV Accuracy")
        axes[0].set_title(
            f"Model Comparison — {self.cv_folds}-Fold CV Accuracy\n"
            "Green = healthy  |  Red = has issues (overfit / underfit / high variance)"
        )
        axes[0].set_ylim(0, 1.05)
        axes[0].axhline(
            UNDERFIT_THRESHOLD, color="orange", linestyle="--", linewidth=1.2,
            label=f"Underfit threshold ({UNDERFIT_THRESHOLD})"
        )
        axes[0].legend(fontsize=9)

        # ── Bottom: Health Score ──
        axes[1].bar(x, health, color="#3498db", alpha=0.85)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(names, rotation=35, ha="right", fontsize=9)
        axes[1].set_ylabel("Health Score")
        axes[1].set_title(
            "Composite Health Score  (penalises overfit gap + high variance + issues)"
        )
        axes[1].set_ylim(0, 1.05)

        # Highlight winner
        best_idx = np.argmax(health)
        axes[1].get_children()[best_idx].set_color("#e67e22")
        axes[1].annotate(
            "★ selected", xy=(best_idx, health[best_idx]),
            xytext=(best_idx, health[best_idx] + 0.03),
            ha="center", fontsize=9, color="#e67e22", fontweight="bold"
        )

        plt.tight_layout()
        plt.savefig("reports/model_comparison.png", dpi=150)
        plt.close()
        logger.info("Model comparison chart    → reports/model_comparison.png ✓")