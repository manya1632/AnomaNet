# AnomaNet — ML Layer
**Stack: Python 3.11 · NetworkX · XGBoost · Isolation Forest · Logistic Regression · PyTorch Geometric · FastAPI · Neo4j · Kafka · Redis · MLflow**

---

## The core idea

Standard fraud monitoring looks at transactions one at a time. It sees ₹9.6L and says: below threshold, pass. It sees a transfer to a new account and says: single transaction, pass.

AnomaNet does not look at transactions. It looks at **the network those transactions create.**

Every time money moves from Account A to Account B, a directed edge is born in a graph. Over thousands of transactions that graph reveals structures completely invisible in a flat table — a ring of accounts passing money in a circle, a single account exploding outward to eight new accounts in 40 minutes, three deposits from the same person all landing just below the regulatory filing limit. These are not transaction anomalies. They are **graph anomalies.** This layer exists to find them.

---

## What is built so far

1. Data Simulator — `data_simulator/`
2. Neo4j Client — `core/graph/neo4j_client.py`
3. Circular Detector — `core/scoring/circular_detector.py`
4. Layering Scorer — `core/scoring/layering_scorer.py`
5. Structuring Scorer — `core/scoring/structuring_scorer.py`
6. Dormancy Scorer — `core/scoring/dormancy_scorer.py`
7. AnomaScore Aggregator — `core/scoring/anoma_score.py`

---

## Part 1 — Data Simulator

### Why it exists

The ML models need labelled training data. Real bank fraud data is confidential, legally restricted, and impossible to obtain for a hackathon. The simulator generates 100,000 synthetic transactions that are statistically realistic enough to train production-quality classifiers — correct distributions of amounts, channels, timings, account types, and Indian banking behaviour — with ground-truth fraud labels embedded.

### The import architecture — why models.py exists

When the simulator was first written, `simulator.py` imported from `layering.py` and `layering.py` imported back from `simulator.py`. Python partially initialises a module before finishing its imports, so this circular reference caused an ImportError at runtime.

The fix: extract everything shared into `models.py`, which imports from nothing inside this package. It is the base of the dependency tree.

```
models.py                    ← imports nothing from this package
    ↑               ↑
simulator.py     scenarios/*.py
```

`models.py` owns: the `Customer`, `Account`, `Transaction` dataclasses, all Indian banking constants (IFSC prefixes, channel probability weights, KYC tier income bands), and all pure helper functions. Every other file in the simulator imports only from here.

### The five fraud scenarios

**Layering (`scenarios/layering.py`)**

Layering is Phase 2 of money laundering: Placement → Layering → Integration. The goal is to obscure the trail by moving money through as many accounts as fast as possible.

The simulator creates clusters of 1 source account + 5–8 mule accounts, each on a distinct IFSC branch. Dirty money (₹30L–₹2Cr) arrives at the source and fans out to all mules within 90 minutes. Each mule holds the money less than 15 minutes before forwarding onward. Timestamps are anchored to 2–5 AM because off-hours activity reduces real-time human review risk.

The critical insight: no single transaction is unusual. The signals are velocity (8 outbound transfers in 1 hour), topology (fan-out from one node to many), and residency time (money arrives and immediately leaves). Invisible in a flat table. Obvious in a graph.

**Circular / Round-tripping (`scenarios/circular.py`)**

Creates rings of 2–7 accounts. Money travels A→B→C→...→A, completing the full cycle within 2–72 hours. Each hop deducts a small fake fee (0.5–2%) so amounts decay slightly across the ring, mimicking real round-tripping where each entity issues a fake invoice to justify the transfer.

In a PostgreSQL transactions table, finding A→B→C→A requires joining the table against itself 3 times with timestamp filters. With 100k rows that query is slow and brittle. In Neo4j this is a single Cypher pattern match running in milliseconds. Johnson's Algorithm then confirms it in NetworkX.

Amount variance is deliberately kept low (±15%) because equal amounts are the signature of money going nowhere real — legitimate business transactions vary wildly in amount.

**Structuring / Smurfing (`scenarios/structuring.py`)**

India's RBI mandates a Cash Transaction Report for cash transactions above ₹10,00,000. Fraudsters avoid this by making multiple deposits each just below the threshold.

The simulator generates 3–7 cash deposits per cluster, each between 90% and 98.5% of a threshold. The spread is not round — real structuring avoids suspiciously round numbers like exactly ₹9,00,000. A smurfing variant (35% of clusters) spreads deposits across multiple branches. Three threshold tiers: ₹10L (primary CTR), ₹5L, ₹2L, weighted 70/20/10.

**Dormant Account Activation (`scenarios/dormant_activation.py`)**

An account silent for 14–24 months suddenly receives ₹50L+ and wires it back out within 2–6 hours. The historical average transaction size for these accounts is ₹5k–₹30k — the inbound is deliberately 10× or more this baseline. The `kyc_recently_updated` flag is embedded in metadata because fraudsters update mobile numbers before using a takeover account so OTPs go to them.

**Profile Mismatch (`scenarios/profile_mismatch_gen.py`)**

A kirana shop owner with declared monthly income ₹40,000 starts receiving SWIFT transfers from offshore BICs totalling ₹32 lakhs in a month. This scenario generates months of realistic small historical transactions first — the baseline Rupali's autoencoder needs. Then the anomalous burst is injected: 5–20 large SWIFT/RTGS transfers totalling 15×–80× the declared monthly income. The channel mismatch alone — rural SAVINGS account receiving international wires — is a strong signal.

### Output files

| File | Contents | Used by |
|---|---|---|
| `transactions.parquet` | 100,000 transactions + fraud labels | Training all classifiers |
| `accounts.parquet` | Account master with KYC fields and dormancy status | Dormancy scorer, feature engineering |
| `customers.parquet` | Customer KYC — name, income, occupation, city | Profile mismatch training |
| `labels.parquet` | Ground truth only — is_fraud, fraud_type, cluster_id | Training / evaluation (separate to prevent label leakage into features) |
| `neo4j_nodes.parquet` | Account + Customer + Branch nodes | `bulk_load_from_simulator()` |
| `neo4j_edges.parquet` | TRANSFERRED_TO + OWNS + BELONGS_TO edges | `bulk_load_from_simulator()` |

The labels file is intentionally separate. If `transactions.parquet` contained both features and labels, a careless training script might include `is_fraud` as an input feature. Keeping them separate forces an explicit merge.

---

## Part 2 — Neo4j Client

**`core/graph/neo4j_client.py`**

Everything the ML layer needs from the graph database lives in this file. No other ML file talks to Neo4j directly.

### Why Neo4j, not PostgreSQL

PostgreSQL is the system of record — every transaction lives there for audit. But fraud detection needs graph queries. "Find all accounts that transacted with accounts that transacted with a known fraud account within 72 hours" is a recursive self-join in SQL that degrades exponentially with each hop. In Neo4j's Cypher it is one pattern match. The graph database also maintains structural properties — degree, clustering, centrality — that are expensive to compute from scratch in SQL but inherent in the graph structure.

### Key functions

`get_subgraph(account_id, hops, hours)` — the main workhorse. Extracts an N-hop neighbourhood (capped at 4 for performance) into a NetworkX DiGraph. Every node carries account attributes; every edge carries amount, timestamp, channel, tx_id. The time filter ensures only recent transactions are included — a cycle from 3 years ago is not a current fraud signal.

`get_cycle_candidates(account_id, max_hops, hours)` — Cypher pre-filter before running Johnson's Algorithm. Neo4j's native pattern matching uses graph indexes and is much faster than running the full algorithm on the entire graph.

`get_historical_avg_amount(account_id, days)` — the dormancy scorer needs to know what "normal" looks like for an account before flagging a transaction as 10× the baseline.

`write_transaction_edge(...)` — called by the Kafka consumer every time a transaction is scored. Uses MERGE so account nodes are created if they don't exist. This is how the live graph grows in real time.

`bulk_load_from_simulator()` — batch-loads the parquet output into Neo4j in groups of 500 rows. Run once after generating the dataset.

---

## Part 3 — The Five Fraud Detectors

### Circular Detector (`core/scoring/circular_detector.py`)

**Theory**

Johnson's Algorithm finds all simple directed cycles in a directed graph. A simple cycle visits no node twice except the start/end node. NetworkX implements this as `nx.simple_cycles()`. We run it on the subgraph extracted from Neo4j, then filter by four fraud criteria:

1. Cycle length 2–7 hops
2. Completed within 72 hours
3. Edge amounts within ±15% variance — round-tripping keeps amounts similar to sustain the fake invoice fiction
4. At least 2 first-time counterparty relationships — new connections in a cycle suggest it was set up specifically for this fraud

**Scoring**: base 0.70 if all four criteria pass. Bonuses: +0.10 for completion under 6 hours, +0.08 for amount variance under 5%, +0.07 for 3 or more first-time edges. Maximum 1.0.

**Two entry points**: `score_circular(account_id)` connects to live Neo4j for production. `score_circular_from_graph(G, account_id)` takes a pre-built NetworkX graph for tests and training — no database needed.

---

### Layering Scorer (`core/scoring/layering_scorer.py`)

**Theory**

Two models in ensemble. Final score = max(both).

**Hard velocity rule** — deterministic:
```
tx_count_1h > 5  AND  total_amount_1h > ₹5,00,000  →  score = 0.80
```
Both conditions required. A single ₹50L RTGS payment trips the amount but not the count — that is legitimate high-value banking. Eight ₹50k UPI payments trip the count but not the amount — that could be payroll. Both firing together means money is moving to many places very fast.

**Isolation Forest** — unsupervised anomaly detection trained on the clean transaction distribution. Builds a forest of random decision trees; anomalous records are isolated near the root (few splits needed) because they sit in sparse regions of the feature space. Output normalised to [0, 1].

The model loads from `core/models/isolation_forest_layering.pkl`. If the file does not exist (before training is run), the scorer logs a warning and runs rule-only. **This is why tests reported "model not found — using rule based." That is correct expected behaviour.** The fallback is intentional so the system is demo-able before training completes.

Contextual bonuses on top: off-hours (+0.05), cross-branch ratio above 70% (+0.06), residency under 5 minutes (+0.07), more than 10 unique counterparties in 24 hours (+0.05).

---

### Structuring Scorer (`core/scoring/structuring_scorer.py`)

**Theory**

Scans all cash and branch-channel transactions from the account in the last 7 days. An amount is "near threshold" if it falls between 85% and 99% of ₹10L, ₹5L, or ₹2L.

**XGBoost classifier** — binary classification trained on engineered features. XGBoost is chosen over logistic regression because the relationship is non-linear: 3 transactions at 95% of threshold is more suspicious than 3 at 86%, but 6 transactions at 86% is more suspicious than 3 at 95%. A decision-tree ensemble captures these interactions naturally.

The `declared_income_ratio` feature is particularly powerful. A merchant processing ₹28L in cash deposits might be legitimate if their declared monthly income is ₹8L. The same deposits from someone declaring ₹40,000/month is almost certainly fraud.

If XGBoost model not trained yet, calibrated rule fires: 3 suspicious transactions → 0.60, 4–5 → 0.70, 6+ → 0.80, with bonuses for multiple branches and tight time clustering.

---

### Dormancy Scorer (`core/scoring/dormancy_scorer.py`)

**Theory**

Two-stage: state machine gates first, logistic regression adjusts.

**Stage 1 — State machine**: account must satisfy both:
1. `is_dormant = True` — no activity for 12+ months (RBI definition)
2. Current transaction > 10× historical average amount

If either fails, maximum partial score of 0.35–0.55 based purely on dormancy duration. This prevents alerts on dormant accounts receiving small legitimate transactions.

**Stage 2 — Logistic regression**: when state machine fires (base 0.75), adjusts based on: dormancy duration in months, how quickly money was wired out after arrival, whether KYC was recently updated (classic takeover signal), amount vs declared income ratio, and KYC risk tier.

Logistic regression loads from `core/models/logistic_dormancy.pkl`. Rule bonuses fire directly if model not trained yet.

---

### AnomaScore Aggregator (`core/scoring/anoma_score.py`)

**Theory**

Calls all five detectors and computes the weighted sum:

```
AnomaScore = 0.25 × layering
           + 0.30 × circular
           + 0.20 × structuring
           + 0.10 × dormancy
           + 0.15 × profile_mismatch
```

**Why these weights**: Circular gets highest (0.30) because Johnson's Algorithm is deterministic — when it finds a valid cycle the false positive rate is very low. Layering is second (0.25) because velocity signals are strong but Isolation Forest occasionally misfires on legitimate high-volume merchants. Profile mismatch is lowest (0.15) because it requires a trained autoencoder — before training it contributes 0.0.

Weights are **runtime-configurable** via the admin panel without redeployment. The only enforcement is they must sum to 1.0.

**Alert thresholds**: standard accounts alert at AnomaScore ≥ 0.65, PEP-tier accounts at ≥ 0.45 (RBI enhanced due diligence).

**Resilience**: every detector call is in try/except. If circular detector crashes (Neo4j timeout), the other four still run and a partial score is published. The system never goes silent because one component failed.

---

## About the "model not found" message

When you ran the tests you saw:
```
WARNING — Isolation Forest model not found — using rule based
WARNING — XGBoost model not found — using rule based
```

This is **correct and intentional**. The training scripts (`train_classifiers.py`) have not been run yet, so no `.pkl` files exist in `core/models/`. The scorers detect this at startup and fall back to calibrated rule-based scoring that produces realistic scores. The system is fully demo-able before training completes. Once training runs, the `.pkl` files are written and the models load automatically next startup.

---

## How to set up

```powershell
# Step 1 — from repo root AnomaNet/
docker-compose up -d

# Wait 60 seconds then check all containers
docker-compose ps
# All should show healthy or running

# Step 2 — from AnomaNet/ml/
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt

# Step 3 — generate training data (run once, ~3-5 minutes)
python -m data_simulator.simulator --output data

# Step 4 — load into Neo4j (run once after Step 3)
python -c "from core.graph.neo4j_client import bulk_load_from_simulator; bulk_load_from_simulator()"
```

---

## How to test everything built

### Tests that need nothing except installed packages

**Test 1 — Circular detector catches a valid fraud cycle**
```powershell
python -c "
import networkx as nx
from datetime import datetime, timezone, timedelta
from core.scoring.circular_detector import score_circular_from_graph

G = nx.DiGraph()
now = datetime.now(tz=timezone.utc)
G.add_edge('A','B', amount=4820000, timestamp=now-timedelta(hours=3), channel='NEFT', tx_id='t1', branch_id='HDFC0001')
G.add_edge('B','C', amount=4790000, timestamp=now-timedelta(hours=2), channel='NEFT', tx_id='t2', branch_id='ICIC0001')
G.add_edge('C','A', amount=4760000, timestamp=now-timedelta(hours=1), channel='RTGS', tx_id='t3', branch_id='SBIN0001')
r = score_circular_from_graph(G, 'A')
print('Detected :', r.cycle_detected)
print('Score    :', r.cycle_score)
print('Path     :', r.cycle_path)
print('Hours    :', r.completion_hours)
"
```
Expected: `Detected: True`, score 0.70–0.95, path `['A','B','C','A']`

---

**Test 2 — Circular detector correctly rejects high-variance cycle**
```powershell
python -c "
import networkx as nx
from datetime import datetime, timezone, timedelta
from core.scoring.circular_detector import score_circular_from_graph

G = nx.DiGraph()
now = datetime.now(tz=timezone.utc)
G.add_edge('A','B', amount=1000000, timestamp=now-timedelta(hours=2), channel='NEFT', tx_id='t1', branch_id='HDFC0001')
G.add_edge('B','A', amount=600000,  timestamp=now-timedelta(hours=1), channel='NEFT', tx_id='t2', branch_id='ICIC0001')
r = score_circular_from_graph(G, 'A')
print('Detected:', r.cycle_detected)
print('Score   :', r.cycle_score)
"
```
Expected: `Detected: False`, score 0.0 — amounts differ 40%, cycle rejected

---

**Test 3 — Layering scorer fires on velocity pattern**
```powershell
python -c "
from core.scoring.layering_scorer import score_layering_from_features
r = score_layering_from_features({
    'tx_count_1h': 8,
    'total_amount_1h': 4500000,
    'unique_counterparties_24h': 7,
    'cross_branch_ratio': 0.85,
    'residency_seconds': 420,
    'is_off_hours': 1,
    'tx_count_24h': 10,
    'total_amount_24h': 5200000,
}, account_id='TEST_LAYER')
print('Score      :', r.layering_score)
print('Rule fired :', r.rule_fired)
"
```
Expected: score ≥ 0.85, rule fired = True

---

**Test 4 — Layering scorer passes a clean account**
```powershell
python -c "
from core.scoring.layering_scorer import score_layering_from_features
r = score_layering_from_features({
    'tx_count_1h': 2,
    'total_amount_1h': 45000,
    'unique_counterparties_24h': 2,
    'cross_branch_ratio': 0.0,
    'residency_seconds': 9999,
    'is_off_hours': 0,
}, account_id='CLEAN')
print('Score      :', r.layering_score)
print('Rule fired :', r.rule_fired)
"
```
Expected: score 0.0, rule fired = False

---

**Test 5 — Structuring scorer catches threshold clustering**
```powershell
python -c "
from core.scoring.structuring_scorer import score_structuring
from datetime import datetime, timezone, timedelta
now = datetime.now(tz=timezone.utc)
txns = [
    {'amount': 960000, 'channel': 'CASH',   'branch_id': 'HDFC0001', 'initiated_at': (now-timedelta(days=3)).isoformat()},
    {'amount': 940000, 'channel': 'CASH',   'branch_id': 'HDFC0002', 'initiated_at': (now-timedelta(days=2)).isoformat()},
    {'amount': 975000, 'channel': 'BRANCH', 'branch_id': 'ICIC0001', 'initiated_at': (now-timedelta(days=1)).isoformat()},
    {'amount': 955000, 'channel': 'CASH',   'branch_id': 'SBIN0001', 'initiated_at': now.isoformat()},
]
r = score_structuring('TEST', txns, declared_monthly_income=40000)
print('Score          :', r.structuring_score)
print('Threshold tier : INR', r.threshold_tier)
print('Suspicious txns:', r.n_suspicious_txns)
print('Aggregate      : INR', r.aggregate_amount)
"
```
Expected: score ≥ 0.65, tier = 1000000, suspicious = 4, aggregate ≈ 3830000

---

**Test 6 — Structuring scorer ignores non-cash transactions**
```powershell
python -c "
from core.scoring.structuring_scorer import score_structuring
from datetime import datetime, timezone, timedelta
now = datetime.now(tz=timezone.utc)
txns = [
    {'amount': 960000, 'channel': 'UPI',  'branch_id': 'HDFC0001', 'initiated_at': (now-timedelta(days=1)).isoformat()},
    {'amount': 940000, 'channel': 'NEFT', 'branch_id': 'HDFC0001', 'initiated_at': now.isoformat()},
]
r = score_structuring('CLEAN', txns, declared_monthly_income=200000)
print('Score:', r.structuring_score)
"
```
Expected: score 0.0 — CTR only applies to cash/branch channels

---

### Tests that need docker-compose up

**Test 7 — Neo4j is reachable**
```powershell
python -c "
from core.graph.neo4j_client import health_check
print('Neo4j healthy:', health_check())
"
```
Expected: `Neo4j healthy: True`

---

**Test 8 — Verify simulator output**
```powershell
python -c "
import pandas as pd
tx = pd.read_parquet('data/transactions.parquet')
print('Total transactions:', len(tx))
print()
print('Fraud breakdown:')
print(tx[tx.is_fraud]['fraud_type'].value_counts())
print()
print('Clean transactions:', len(tx[~tx.is_fraud]))
"
```
Expected: 100,000 total, ~1,000 per fraud type, ~95,000 clean

---

### Tests that need docker-compose up AND bulk_load_from_simulator() completed

**Test 9 — Subgraph extraction from live graph**
```powershell
python -c "
import pandas as pd
from core.graph.neo4j_client import get_subgraph

tx = pd.read_parquet('data/transactions.parquet')
account_id = tx[tx.fraud_type == 'CIRCULAR'].iloc[0]['source_account_id']
print('Account:', account_id)

G = get_subgraph(account_id, hops=3, hours=720)
print('Nodes:', G.number_of_nodes())
print('Edges:', G.number_of_edges())
"
```
Expected: nodes > 1, edges > 0

---

**Test 10 — Circular detector on a real fraud account from dataset**
```powershell
python -c "
import pandas as pd
from core.graph.neo4j_client import get_subgraph
from core.scoring.circular_detector import score_circular_from_graph

tx = pd.read_parquet('data/transactions.parquet')
account_id = tx[tx.fraud_type == 'CIRCULAR'].iloc[0]['source_account_id']
print('Account:', account_id)

G = get_subgraph(account_id, hops=7, hours=720)
r = score_circular_from_graph(G, account_id)
print('Detected:', r.cycle_detected)
print('Score   :', r.cycle_score)
print('Path    :', r.cycle_path)
"
```
Expected: `Detected: True`, score above 0.70

---

## What is not built yet

| File | Purpose | What it unblocks |
|---|---|---|
| `ml/interfaces.py` | Stub for Rupali's 3 functions with safe fallback returns | `anoma_score.py` runs without crashing |
| `training/train_classifiers.py` | Trains Isolation Forest, XGBoost, logistic regression on 100k | Models load instead of rule fallback |
| `core/gnn/graphsage_encoder.py` | 3-layer GraphSAGE, 128-dim embeddings per account | GNN-enhanced scoring |
| `training/train_gnn.py` | Trains GraphSAGE semi-supervised, MLflow tracking | GNN model available |
| `core/main.py` | FastAPI app — POST /ml/score, GET /ml/health | HTTP endpoint live |
| `core/kafka/consumer.py` | Reads ml.scoring.queue, runs pipeline, publishes AlertEvent | Full end-to-end pipeline |

---

## Directory structure

```
ml/
├── README.md
├── requirements.txt
├── data/                           ← gitignored — regenerate with simulator
│   ├── transactions.parquet
│   ├── accounts.parquet
│   ├── customers.parquet
│   ├── labels.parquet
│   ├── neo4j_nodes.parquet
│   └── neo4j_edges.parquet
│
├── data_simulator/
│   ├── models.py                   ← shared types + helpers (no circular risk)
│   ├── simulator.py                ← orchestrator, generates 100k
│   └── scenarios/
│       ├── layering.py
│       ├── circular.py
│       ├── structuring.py
│       ├── dormant_activation.py
│       └── profile_mismatch_gen.py
│
├── core/
│   ├── graph/
│   │   └── neo4j_client.py         ← all Neo4j interactions for ML layer
│   ├── scoring/
│   │   ├── circular_detector.py    ← Johnson's Algorithm cycle detection
│   │   ├── layering_scorer.py      ← Isolation Forest + hard velocity rule
│   │   ├── structuring_scorer.py   ← XGBoost threshold clustering classifier
│   │   ├── dormancy_scorer.py      ← state machine + logistic regression
│   │   └── anoma_score.py          ← weighted composite aggregator
│   ├── gnn/                        ← not built yet
│   ├── kafka/                      ← not built yet
│   └── models/                     ← .pkl files written here after training
│
├── training/                       ← not built yet
├── modules/                        ← Rupali's modules (interface boundary)
└── shared/                         ← Rupali's Redis feature store
```