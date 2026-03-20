"""
ml/training/train_gnn.py

Semi-supervised GraphSAGE training on the full account graph.

Usage:
  python -m training.train_gnn
  python -m training.train_gnn --epochs 100 --no-mlflow
"""

from __future__ import annotations

import argparse
import logging
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

MODELS_DIR   = Path(__file__).parent.parent / "core" / "models"
MLFLOW_URI   = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
RANDOM_STATE = 42

torch.manual_seed(RANDOM_STATE)
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

try:
    from torch_geometric.data import Data
    from core.gnn.graphsage_encoder import (
        GraphSAGEEncoder, build_node_features, build_edge_index,
        save_model, NODE_FEATURE_DIM,
    )
    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False
    log.error("torch_geometric not installed — pip install torch-geometric")


def build_training_graph(data_dir: str = "data"):
    d = Path(data_dir)
    log.info("Loading data from %s...", d.resolve())

    tx     = pd.read_parquet(d / "transactions.parquet")
    acct   = pd.read_parquet(d / "accounts.parquet")

    # Per-account velocity features
    tx_agg = tx.groupby("source_account_id").agg(
        tx_count_total  = ("id",             "count"),
        avg_amount      = ("amount",          "mean"),
        unique_cp       = ("dest_account_id", "nunique"),
        unique_branches = ("branch_id",       "nunique"),
    ).reset_index().rename(columns={"source_account_id": "id"})
    tx_agg["cross_branch_ratio"] = (
        tx_agg["unique_branches"] / tx_agg["tx_count_total"].clip(lower=1)
    )

    acct_m = acct.merge(tx_agg, on="id", how="left")
    for col in ["tx_count_total", "avg_amount", "cross_branch_ratio", "unique_cp"]:
        acct_m[col] = acct_m[col].fillna(0)

    account_ids   = acct_m["id"].tolist()
    account_attrs = {}
    for _, row in acct_m.iterrows():
        account_attrs[row["id"]] = {
            "tx_count_total":            row.get("tx_count_total", 0),
            "avg_amount":                row.get("avg_amount", 0),
            "is_dormant":                bool(row.get("is_dormant", False)),
            "kyc_risk_tier":             row.get("kyc_risk_tier", "LOW"),
            "cross_branch_ratio":        row.get("cross_branch_ratio", 0),
            "off_hours_ratio":           0.0,
            "unique_counterparties_24h": row.get("unique_cp", 0),
            "account_type":              row.get("account_type", "SAVINGS"),
            "anoma_score":               0.0,
            "days_since_last_tx":        0.0,
        }

    edges      = list(zip(tx["source_account_id"].tolist(), tx["dest_account_id"].tolist()))
    x          = build_node_features(account_ids, account_attrs)
    edge_index = build_edge_index(account_ids, edges)

    log.info("Graph: %d nodes | %d edges", len(account_ids), edge_index.shape[1])

    # Labels
    fraud_accts = set(tx[tx["is_fraud"] == True]["source_account_id"].tolist())
    id_to_idx   = {aid: i for i, aid in enumerate(account_ids)}
    y = torch.zeros(len(account_ids), dtype=torch.long)
    for aid in fraud_accts:
        if aid in id_to_idx:
            y[id_to_idx[aid]] = 1

    n_fraud = int(y.sum())
    log.info("Labels: %d fraud | %d clean", n_fraud, len(account_ids) - n_fraud)

    # Balanced split
    fraud_idx = torch.where(y == 1)[0].tolist()
    clean_idx = torch.where(y == 0)[0].tolist()
    clean_sample = random.sample(clean_idx, min(len(clean_idx), n_fraud * 3))
    all_idx      = fraud_idx + clean_sample

    train_idx, temp_idx = train_test_split(
        all_idx, test_size=0.4, random_state=RANDOM_STATE,
        stratify=[y[i].item() for i in all_idx],
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, random_state=RANDOM_STATE,
        stratify=[y[i].item() for i in temp_idx],
    )

    train_mask = torch.zeros(len(account_ids), dtype=torch.bool)
    val_mask   = torch.zeros(len(account_ids), dtype=torch.bool)
    test_mask  = torch.zeros(len(account_ids), dtype=torch.bool)
    for i in train_idx: train_mask[i] = True
    for i in val_idx:   val_mask[i]   = True
    for i in test_idx:  test_mask[i]  = True

    data = Data(x=x, edge_index=edge_index, y=y,
                train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)

    stats = {
        "n_nodes": len(account_ids), "n_edges": edge_index.shape[1],
        "n_fraud": n_fraud, "n_train": int(train_mask.sum()),
        "n_val": int(val_mask.sum()), "n_test": int(test_mask.sum()),
    }
    return data, account_ids, stats


def train_epoch(model, data, optimizer, class_weights):
    model.train()
    optimizer.zero_grad()
    _, logits = model(data.x, data.edge_index)
    loss = F.cross_entropy(logits[data.train_mask], data.y[data.train_mask],
                           weight=class_weights)
    loss.backward()
    optimizer.step()
    return float(loss)


@torch.no_grad()
def evaluate(model, data, mask):
    model.eval()
    _, logits = model(data.x, data.edge_index)
    preds  = logits[mask].argmax(dim=1).cpu().numpy()
    probs  = F.softmax(logits[mask], dim=1)[:, 1].cpu().numpy()
    labels = data.y[mask].cpu().numpy()
    f1     = f1_score(labels, preds, zero_division=0)
    try:
        auc = roc_auc_score(labels, probs)
    except Exception:
        auc = 0.0
    return {"f1": f1, "auc": auc, "preds": preds, "labels": labels}


def main(data_dir="data", epochs=200, lr=0.005, use_mlflow=True):
    if not PYG_AVAILABLE:
        log.error("torch_geometric not installed. Cannot train GNN.")
        return

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    data, account_ids, stats = build_training_graph(data_dir)
    data = data.to(device)
    log.info("Stats: %s", stats)

    n_fraud = float((data.y == 1).sum())
    n_clean = float((data.y == 0).sum())
    class_weights = torch.tensor([1.0, n_clean / max(n_fraud, 1)],
                                  dtype=torch.float32).to(device)

    model     = GraphSAGEEncoder(in_channels=NODE_FEATURE_DIM).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=20, min_lr=1e-5)

    best_val_f1, best_state, patience_cnt = 0.0, None, 0
    EARLY_STOP = 40

    for epoch in range(1, epochs + 1):
        loss = train_epoch(model, data, optimizer, class_weights)
        val  = evaluate(model, data, data.val_mask)
        scheduler.step(val["f1"])

        if val["f1"] > best_val_f1:
            best_val_f1  = val["f1"]
            best_state   = {k: v.clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1

        if epoch % 20 == 0 or epoch == 1:
            log.info("Epoch %3d | loss=%.4f | val_f1=%.3f | val_auc=%.3f | best=%.3f",
                     epoch, loss, val["f1"], val["auc"], best_val_f1)

        if patience_cnt >= EARLY_STOP:
            log.info("Early stop at epoch %d", epoch)
            break

    if best_state:
        model.load_state_dict(best_state)

    test = evaluate(model, data, data.test_mask)
    log.info("=" * 50)
    log.info("TEST  F1=%.3f | AUC=%.3f", test["f1"], test["auc"])
    log.info("=" * 50)
    print(classification_report(test["labels"], test["preds"],
                                 target_names=["clean", "fraud"]))

    if test["f1"] < 0.75:
        log.warning("F1=%.3f below 0.80 target — try more epochs or lr tuning", test["f1"])

    model_path = str(MODELS_DIR / "graphsage_encoder.pt")
    save_model(model, model_path)

    if use_mlflow:
        try:
            import mlflow
            mlflow.set_tracking_uri(MLFLOW_URI)
            mlflow.set_experiment("AnomaNet_GNN")
            with mlflow.start_run(run_name="GraphSAGE_Encoder"):
                mlflow.log_params({"epochs": epochs, "lr": lr, "hidden_dim": 64,
                                   "out_dim": 128, "dropout": 0.3, **stats})
                mlflow.log_metrics({"best_val_f1": best_val_f1,
                                    "test_f1": test["f1"], "test_auc": test["auc"]})
                mlflow.log_artifact(model_path)
        except Exception as e:
            log.warning("MLflow failed: %s", e)

    log.info("GNN training complete → %s", model_path)
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",      default="data")
    parser.add_argument("--epochs",    type=int,   default=200)
    parser.add_argument("--lr",        type=float, default=0.005)
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()
    main(data_dir=args.data, epochs=args.epochs, lr=args.lr,
         use_mlflow=not args.no_mlflow)