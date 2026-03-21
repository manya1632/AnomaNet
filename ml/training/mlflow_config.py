"""
ml/training/mlflow_config.py

Central MLflow configuration for all AnomaNet experiments.
Import this at the top of any training script to get a consistent
experiment setup with the right tracking URI, tags, and naming.

Usage:
    from training.mlflow_config import setup_mlflow, log_model_summary

    setup_mlflow("IsolationForest_Layering")
    # ... training ...
    log_model_summary(model, metrics, params)
"""

from __future__ import annotations

import logging
import os
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

MLFLOW_TRACKING_URI  = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME      = "AnomaNet_Fraud_Detection"
MODELS_DIR           = Path(__file__).parent.parent / "core" / "models"

# Detector → experiment run name mapping
DETECTOR_RUN_NAMES = {
    "layering":        "IsolationForest_Layering",
    "structuring":     "XGBoost_Structuring",
    "dormancy":        "LogisticRegression_Dormancy",
    "gnn":             "GraphSAGE_Encoder",
    "classifiers_all": "All_Classifiers",
}


# ── Setup ─────────────────────────────────────────────────────────────────────

def setup_mlflow(run_name: Optional[str] = None) -> bool:
    """
    Configure MLflow tracking URI and set the experiment.
    Returns True if MLflow is available and connected, False otherwise.

    Call this once at the start of a training script:
        if not setup_mlflow("IsolationForest_Layering"):
            log.warning("MLflow not available — metrics will only be logged locally")
    """
    try:
        import mlflow
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(EXPERIMENT_NAME)

        if run_name:
            mlflow.start_run(
                run_name=run_name,
                tags={
                    "project":   "AnomaNet",
                    "owner":     "Muskan",
                    "python":    platform.python_version(),
                    "timestamp": datetime.now(tz=timezone.utc).isoformat(),
                }
            )

        log.info("MLflow configured → %s | experiment: %s", MLFLOW_TRACKING_URI, EXPERIMENT_NAME)
        return True

    except Exception as e:
        log.warning("MLflow setup failed: %s — training will proceed without tracking", e)
        return False


def end_run():
    """End the current MLflow run cleanly."""
    try:
        import mlflow
        mlflow.end_run()
    except Exception:
        pass


# ── Logging helpers ───────────────────────────────────────────────────────────

def log_params(params: dict):
    """Log hyperparameters to the active MLflow run."""
    try:
        import mlflow
        mlflow.log_params(params)
    except Exception as e:
        log.debug("MLflow log_params failed: %s", e)


def log_metrics(metrics: dict, step: Optional[int] = None):
    """Log evaluation metrics to the active MLflow run."""
    try:
        import mlflow
        mlflow.log_metrics(metrics, step=step)
    except Exception as e:
        log.debug("MLflow log_metrics failed: %s", e)


def log_model_sklearn(model, artifact_name: str):
    """Log a scikit-learn model artifact."""
    try:
        import mlflow.sklearn
        mlflow.sklearn.log_model(model, artifact_name)
        log.info("MLflow: logged sklearn model as '%s'", artifact_name)
    except Exception as e:
        log.debug("MLflow log_model_sklearn failed: %s", e)


def log_model_pytorch(model, artifact_name: str):
    """Log a PyTorch model artifact."""
    try:
        import mlflow.pytorch
        mlflow.pytorch.log_model(model, artifact_name)
        log.info("MLflow: logged pytorch model as '%s'", artifact_name)
    except Exception as e:
        log.debug("MLflow log_model_pytorch failed: %s", e)


def log_model_summary(
    model,
    metrics: dict,
    params: dict,
    artifact_name: Optional[str] = None,
    is_pytorch: bool = False,
):
    """
    One-shot: log params + metrics + model artifact in a single call.
    Use this at the end of every training run.

    Example:
        log_model_summary(
            model,
            metrics={"f1": 0.93, "auc": 0.98, "precision": 0.89, "recall": 0.98},
            params={"C": 1.0, "class_weight": "balanced", "detector": "dormancy"},
            artifact_name="logistic_dormancy",
        )
    """
    log_params(params)
    log_metrics(metrics)
    if artifact_name:
        if is_pytorch:
            log_model_pytorch(model, artifact_name)
        else:
            log_model_sklearn(model, artifact_name)


# ── Experiment summary ────────────────────────────────────────────────────────

def print_experiment_summary():
    """
    Print a table of all runs in the AnomaNet experiment.
    Useful to check what's been trained before running again.

    Usage:
        python -c "from training.mlflow_config import print_experiment_summary; print_experiment_summary()"
    """
    try:
        import mlflow
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = mlflow.tracking.MlflowClient()

        experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
        if experiment is None:
            print(f"No experiment named '{EXPERIMENT_NAME}' found.")
            print(f"Run a training script first: python -m training.train_classifiers --no-mlflow")
            return

        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"],
        )

        if not runs:
            print("No runs found in experiment.")
            return

        print(f"\n{'='*70}")
        print(f"Experiment: {EXPERIMENT_NAME}")
        print(f"Tracking:   {MLFLOW_TRACKING_URI}")
        print(f"{'='*70}")
        print(f"{'Run Name':<35} {'F1':>6} {'AUC':>6} {'Status':<12}")
        print(f"{'-'*70}")

        for run in runs:
            name    = run.data.tags.get("mlflow.runName", "unnamed")[:34]
            f1      = run.data.metrics.get("f1",  None)
            auc     = run.data.metrics.get("auc", None)
            status  = run.info.status

            f1_str  = f"{f1:.3f}"  if f1  is not None else "  —  "
            auc_str = f"{auc:.3f}" if auc is not None else "  —  "

            print(f"{name:<35} {f1_str:>6} {auc_str:>6} {status:<12}")

        print(f"{'='*70}\n")

    except Exception as e:
        print(f"Could not fetch MLflow runs: {e}")
        print("Is MLflow running? Start with: docker-compose up -d mlflow")


# ── Local model registry (when MLflow not available) ─────────────────────────

def list_local_models() -> dict:
    """
    List all trained model files in core/models/.
    Used when MLflow is not available.

    Returns dict of {model_name: {path, size_kb, trained_at}}
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    models = {}

    for path in sorted(MODELS_DIR.glob("*.pkl")) :
        stat = path.stat()
        models[path.stem] = {
            "path":       str(path),
            "size_kb":    round(stat.st_size / 1024, 1),
            "trained_at": datetime.fromtimestamp(
                stat.st_mtime, tz=timezone.utc
            ).isoformat(),
        }

    for path in sorted(MODELS_DIR.glob("*.pt")):
        stat = path.stat()
        models[path.stem] = {
            "path":       str(path),
            "size_kb":    round(stat.st_size / 1024, 1),
            "trained_at": datetime.fromtimestamp(
                stat.st_mtime, tz=timezone.utc
            ).isoformat(),
        }

    return models


def print_local_models():
    """Print all trained models in core/models/."""
    models = list_local_models()
    if not models:
        print(f"No models found in {MODELS_DIR}")
        print("Run: python -m training.train_classifiers --no-mlflow")
        return

    print(f"\n{'='*60}")
    print(f"Trained models in {MODELS_DIR.name}/")
    print(f"{'='*60}")
    print(f"{'Model':<40} {'Size':>8} {'Trained At'}")
    print(f"{'-'*60}")
    for name, info in models.items():
        print(f"{name:<40} {info['size_kb']:>6}KB  {info['trained_at'][:19]}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    print_local_models()
    print_experiment_summary()