"""
ml/core/gnn/graphsage_encoder.py

GraphSAGE (Hamilton et al., 2017) — 3-layer Graph Neural Network.

Input:  10 account-level features per node
        + 2-hop neighbourhood structure from Neo4j subgraph

Output: 128-dimensional embedding per account that captures
        structural position in the fraud network.

Why GNN over tabular ML:
  Account A looks clean in isolation. But if A's counterparties
  B and C are each connected to 5 known fraud accounts, A's GNN
  embedding reflects this structural proximity. Tabular ML is
  completely blind to this.

Architecture:
  SAGEConv(10 → 64)   → BatchNorm → ReLU → Dropout(0.3)
  SAGEConv(64 → 128)  → BatchNorm → ReLU → Dropout(0.3)
  SAGEConv(128 → 128) → BatchNorm → L2-normalise
  Linear(128 → 2)     → fraud classification head (training only)

Inductive: works on accounts never seen during training.
No retraining needed when new accounts appear.
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)

try:
    from torch_geometric.nn import SAGEConv
    from torch_geometric.data import Data
    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False
    log.warning("torch_geometric not available — GNN encoder disabled, rule-based scoring only")


# ── Feature definitions ───────────────────────────────────────────────────────

NODE_FEATURE_NAMES = [
    "tx_count_normalised",
    "avg_amount_normalised",
    "is_dormant",
    "kyc_risk_encoded",
    "cross_branch_ratio",
    "off_hours_ratio",
    "unique_counterparties_norm",
    "account_type_encoded",
    "anoma_score",
    "days_since_last_tx_norm",
]

NODE_FEATURE_DIM = len(NODE_FEATURE_NAMES)   # 10
EMBEDDING_DIM    = 128

KYC_ENCODING  = {"LOW": 0.0, "MEDIUM": 0.333, "HIGH": 0.667, "PEP": 1.0}
ACCT_ENCODING = {
    "SAVINGS": 0.0, "CURRENT": 0.2, "OD": 0.4,
    "LOAN": 0.6, "NRE": 0.8, "NRO": 1.0,
}


# ── Model ─────────────────────────────────────────────────────────────────────

class GraphSAGEEncoder(nn.Module):
    """
    3-layer GraphSAGE with mean aggregation.
    Produces L2-normalised 128-dim account embeddings.
    """

    def __init__(
        self,
        in_channels: int   = NODE_FEATURE_DIM,
        hidden_dim:  int   = 64,
        out_dim:     int   = EMBEDDING_DIM,
        dropout:     float = 0.3,
        num_classes: int   = 2,
    ):
        super().__init__()
        if not PYG_AVAILABLE:
            raise ImportError("torch_geometric required. pip install torch-geometric")

        self.dropout = dropout

        self.conv1 = SAGEConv(in_channels, hidden_dim, aggr="mean")
        self.conv2 = SAGEConv(hidden_dim,  out_dim,    aggr="mean")
        self.conv3 = SAGEConv(out_dim,     out_dim,    aggr="mean")

        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(out_dim)
        self.bn3 = nn.BatchNorm1d(out_dim)

        self.classifier = nn.Linear(out_dim, num_classes)

        log.info("GraphSAGEEncoder | %d → 64 → 128 → 128 | classes=%d",
                 in_channels, num_classes)

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward through 3 SAGE layers.
        Returns L2-normalised 128-dim embeddings [N, 128].
        """
        # Layer 1
        h = self.conv1(x, edge_index)
        h = self.bn1(h)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        # Layer 2
        h = self.conv2(h, edge_index)
        h = self.bn2(h)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        # Layer 3
        h = self.conv3(h, edge_index)
        h = self.bn3(h)

        # L2 normalise — embeddings on unit hypersphere
        # cosine similarity = dot product, simplifies downstream use
        return F.normalize(h, p=2, dim=1)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (embeddings [N,128], logits [N,2])"""
        emb    = self.encode(x, edge_index)
        logits = self.classifier(emb)
        return emb, logits

    def get_embedding(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Inference only — returns embeddings, no grad."""
        self.eval()
        with torch.no_grad():
            return self.encode(x, edge_index)


# ── Feature extraction ────────────────────────────────────────────────────────

def build_node_features(
    account_ids:   list[str],
    account_attrs: dict[str, dict],
) -> torch.Tensor:
    """
    Build [N, 10] feature matrix from account attribute dicts.
    Attributes come from Neo4j node properties merged with Redis features.
    """
    rows = []
    for acct_id in account_ids:
        a = account_attrs.get(acct_id, {})

        row = [
            math.log1p(float(a.get("tx_count_total",           0) or 0)) / 10.0,
            math.log1p(float(a.get("avg_amount",                0) or 0)) / 15.0,
            float(bool(a.get("is_dormant",                  False))),
            KYC_ENCODING.get(str(a.get("kyc_risk_tier",     "LOW")), 0.0),
            min(max(float(a.get("cross_branch_ratio",           0) or 0), 0.0), 1.0),
            min(max(float(a.get("off_hours_ratio",              0) or 0), 0.0), 1.0),
            math.log1p(float(a.get("unique_counterparties_24h", 0) or 0)) / 8.0,
            ACCT_ENCODING.get(str(a.get("account_type",  "SAVINGS")), 0.0),
            min(max(float(a.get("anoma_score",                  0) or 0), 0.0), 1.0),
            min(float(a.get("days_since_last_tx",               0) or 0) / 365.0, 1.0),
        ]
        rows.append(row)

    return torch.tensor(rows, dtype=torch.float32)


def build_edge_index(
    account_ids: list[str],
    edges: list[tuple[str, str]],
) -> torch.Tensor:
    """Build [2, E] edge_index from (src_id, dst_id) pairs."""
    id_to_idx = {acct_id: i for i, acct_id in enumerate(account_ids)}
    src_list, dst_list = [], []

    for src_id, dst_id in edges:
        if src_id in id_to_idx and dst_id in id_to_idx:
            src_list.append(id_to_idx[src_id])
            dst_list.append(id_to_idx[dst_id])

    if not src_list:
        return torch.zeros((2, 0), dtype=torch.long)

    return torch.tensor([src_list, dst_list], dtype=torch.long)


def networkx_to_pyg(G, fraud_labels: Optional[dict] = None):
    """
    Convert a NetworkX DiGraph (from get_subgraph) to PyG Data object.

    Args:
        G:            nx.DiGraph with node and edge attributes
        fraud_labels: optional {account_id: 0/1} for training

    Returns:
        (data, account_ids) where account_ids[i] = account ID for node i
    """
    if not PYG_AVAILABLE:
        raise ImportError("torch_geometric required")

    account_ids   = list(G.nodes())
    account_attrs = {node: G.nodes[node] for node in account_ids}
    edges         = list(G.edges())

    x          = build_node_features(account_ids, account_attrs)
    edge_index = build_edge_index(account_ids, edges)

    data           = Data(x=x, edge_index=edge_index)
    data.num_nodes = len(account_ids)

    if fraud_labels is not None:
        y = torch.tensor(
            [fraud_labels.get(acct_id, -1) for acct_id in account_ids],
            dtype=torch.long,
        )
        data.y          = y
        data.train_mask = (y >= 0)

    return data, account_ids


# ── Persistence ───────────────────────────────────────────────────────────────

def save_model(model: GraphSAGEEncoder, path: str):
    torch.save({
        "model_state_dict": model.state_dict(),
        "in_channels":      model.conv1.in_channels,
        "hidden_dim":       model.conv1.out_channels,
        "out_dim":          model.conv2.out_channels,
        "dropout":          model.dropout,
    }, path)
    log.info("GraphSAGE saved → %s", path)


def load_model(path: str) -> GraphSAGEEncoder:
    if not PYG_AVAILABLE:
        raise ImportError("torch_geometric required")

    ckpt  = torch.load(path, map_location="cpu")
    model = GraphSAGEEncoder(
        in_channels = ckpt["in_channels"],
        hidden_dim  = ckpt["hidden_dim"],
        out_dim     = ckpt["out_dim"],
        dropout     = ckpt["dropout"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    log.info("GraphSAGE loaded from %s", path)
    return model