# AnomaNet — ML Layer
**Stack: Python 3.11 · NetworkX · XGBoost · Isolation Forest · Logistic Regression · PyTorch Geometric · FastAPI · Neo4j · Kafka · Redis · MLflow**

---

## The core idea

Standard fraud monitoring looks at transactions one at a time. It sees ₹9.6L and says: below threshold, pass. It sees a transfer to a new account and says: single transaction, pass.

AnomaNet does not look at transactions. It looks at **the network those transactions create.**

Every time money moves from Account A to Account B, a directed edge is born in a graph. Over thousands of transactions that graph reveals structures completely invisible in a flat table — a ring of accounts passing money in a circle, a single account exploding outward to eight new accounts in 40 minutes, three deposits from the same person all landing just below the regulatory filing limit. These are not transaction anomalies. They are **graph anomalies.** This layer exists to find them.

---

## What is built

| File | Status |
|---|---|
| `data_simulator/models.py` | ✅ Complete |
| `data_simulator/simulator.py` | ✅ Complete — 100k dataset generated |
| `data_simulator/scenarios/*.py` | ✅ Complete — all 5 fraud typologies |
| `core/graph/neo4j_client.py` | ✅ Complete |
| `core/scoring/circular_detector.py` | ✅ Complete |
| `core/scoring/layering_scorer.py` | ✅ Complete |
| `core/scoring/structuring_scorer.py` | ✅ Complete |
| `core/scoring/dormancy_scorer.py` | ✅ Complete |
| `core/scoring/anoma_score.py` | ✅ Complete |
| `interfaces.py` | ✅ Complete |
| `training/train_classifiers.py` | ✅ Complete — all 3 models trained |
| `core/models/*.pkl` | ✅ Trained and saved |
| `core/main.py` | ✅ Complete — FastAPI live on port 8000 |
| `core/kafka/consumer.py` | ✅ Complete |
| `core/gnn/graphsage_encoder.py` | 🔲 Not built yet (P1 — impressive but not blocking) |
| `training/train_gnn.py` | 🔲 Not built yet |

---

## How a transaction flows through the pipeline

```
Bank transaction initiated
        │
        ▼
Kafka: raw.transactions
        │
        ▼
Ratnesh's transaction-service
  - Enriches with account KYC from PostgreSQL
  - Writes transaction record to PostgreSQL
  - Writes TRANSFERRED_TO edge to Neo4j
  - Publishes EnrichmentEvent to scoring.queue
        │
        ▼
Muskan's Kafka consumer (core/kafka/consumer.py)
        │
        ├── Write edge to Neo4j (live graph update)
        │
        ├── get_rolling_features(account_id)   ← Rupali's Redis store
        │     tx_count_1h, total_amount_24h,
        │     unique_counterparties, cross_branch_ratio ...
        │
        ├── compute_residency()                ← Neo4j query
        │     time between last inbound and this outbound
        │
        ├── get_recent_transactions()          ← Neo4j query
        │     last 7 days for structuring scorer
        │
        ├── compute_anoma_score()              ← 5 detectors in parallel
        │     circular_detector   → cycle_score
        │     layering_scorer     → layering_score
        │     structuring_scorer  → structuring_score
        │     dormancy_scorer     → dormancy_score
        │     score_profile_mismatch() ← Rupali's autoencoder
        │
        │   AnomaScore = 0.25×layering + 0.30×circular
        │              + 0.20×structuring + 0.10×dormancy
        │              + 0.15×profile_mismatch
        │
        ├── update_anoma_score() in Neo4j      ← node colour in D3 graph
        │
        └── If AnomaScore ≥ threshold
              publish AlertEvent → Kafka: alerts.generated
                        │
                        ▼
              Ratnesh's alert-service
              → WebSocket push → Manya's dashboard
```

---

## Part 1 — Data Simulator

### Why it exists

The ML models need labelled training data. Real bank fraud data is confidential and impossible to obtain for a hackathon. The simulator generates 100,000 synthetic transactions with realistic Indian banking statistics and ground-truth fraud labels embedded.

### The import architecture — why models.py exists

`simulator.py` originally imported from `layering.py` and `layering.py` imported back from `simulator.py`. Python partially initialises a module before finishing imports, causing an ImportError. The fix: extract all shared types and helpers into `models.py` which imports nothing from this package.

```
models.py                    ← imports nothing from this package
    ↑               ↑
simulator.py     scenarios/*.py
```

### The five fraud scenarios

**Layering** — 1 source account fans out to 5–8 mule accounts within 90 minutes. Each mule holds money less than 15 minutes before forwarding. Timestamps biased to 2–5 AM. Cross-branch movement forced. Signals: velocity, fan-out degree, residency time, off-hours activity.

**Circular** — rings of 2–7 accounts. A→B→C→A completes within 2–72 hours. Each hop deducts 0.5–2% fake fee so amounts decay. Invisible in SQL. Trivially a directed cycle in Neo4j. Signals: Johnson's Algorithm finds it in milliseconds.

**Structuring** — 3–7 cash deposits each between 90%–98.5% of a CTR threshold (₹10L/₹5L/₹2L) within 7 days. 35% smurfing variant uses multiple branches. Signals: amount clustering, aggregate vs threshold, branch spread.

**Dormant Activation** — accounts silent for 14–24 months. Large inbound (≥₹50L, ≥10× historical average). Money wired out within 2–6 hours. `kyc_recently_updated` flag in metadata. Signals: dormancy duration, inbound/outbound speed, KYC update flag.

**Profile Mismatch** — kirana shop owners with ₹40k declared income receiving SWIFT wires from offshore BICs. Months of small historical transactions generated first so the autoencoder has a baseline. Then 15×–80× declared income burst injected. Signals: channel mismatch, volume vs declared income, new counterparty network.

### Output files

| File | Contents |
|---|---|
| `transactions.parquet` | 100,000 transactions with fraud labels |
| `accounts.parquet` | Account master with KYC and dormancy fields |
| `customers.parquet` | Customer KYC — income, occupation, city |
| `labels.parquet` | Ground truth only — kept separate to prevent feature leakage |
| `neo4j_nodes.parquet` | Account + Customer + Branch nodes for bulk import |
| `neo4j_edges.parquet` | TRANSFERRED_TO + OWNS + BELONGS_TO edges |

---

## Part 2 — Neo4j Client

**`core/graph/neo4j_client.py`** — everything the ML layer needs from the graph database in one file. No other ML file talks to Neo4j directly.

### Why Neo4j not PostgreSQL

PostgreSQL stores transactions as rows. Finding A→B→C→A requires joining the table against itself 3 times with timestamp filters — exponentially slower with each hop. Neo4j stores transactions as directed edges. Finding all cycles of length 2–7 is a single Cypher pattern match that runs in milliseconds regardless of dataset size.

### Key functions

`get_subgraph(account_id, hops, hours)` — extracts an N-hop neighbourhood as a NetworkX DiGraph with full node and edge attributes. Every detector calls this.

`get_cycle_candidates(account_id, max_hops, hours)` — Cypher pre-filter before Johnson's Algorithm. Uses graph indexes for speed.

`get_historical_avg_amount(account_id)` — dormancy scorer needs the historical baseline to check if current transaction is 10× normal.

`write_transaction_edge(...)` — called by Kafka consumer to grow the live graph in real time.

`bulk_load_from_simulator()` — loads all 100k parquet rows into Neo4j in batches of 500. Run once after generating the dataset.

---

## Part 3 — The Five Fraud Detectors

### Circular Detector

**Algorithm**: Johnson's Algorithm via `nx.simple_cycles()` on the Neo4j subgraph. Finds all simple directed cycles.

**Validation criteria** (all four must pass):
1. Cycle length 2–7 hops
2. Completed within 72 hours
3. Edge amounts within ±15% variance
4. At least 2 first-time counterparty relationships

**Scoring**: base 0.70 if all pass. +0.10 for completion under 6 hours. +0.08 for variance under 5%. +0.07 for 3+ first-time edges.

**Two entry points**: `score_circular(account_id)` for production (uses Neo4j). `score_circular_from_graph(G, account_id)` for tests (no database needed).

---

### Layering Scorer

**Two models, take the maximum:**

Hard velocity rule — deterministic: `tx_count_1h > 5 AND total_amount_1h > ₹5L → score = 0.80`. Both conditions required to avoid false positives on single large transfers or payroll runs.

Isolation Forest — unsupervised anomaly detection trained on clean account velocity features. Builds random decision trees; anomalous accounts isolate near the root. Loaded from `core/models/isolation_forest_layering.pkl`.

**Training result**: F1=0.025, AUC=0.853. Low F1 is because the default threshold is conservative. AUC 0.853 means the model correctly ranks fraud accounts above clean ones — it just needs a calibrated threshold. The hard rule compensates in practice.

**Contextual bonuses**: off-hours (+0.05), cross-branch ratio >70% (+0.06), residency <5 min (+0.07), >10 unique counterparties (+0.05).

---

### Structuring Scorer

**Model**: XGBoost binary classifier. Trained on all accounts with cash/branch transactions — clean accounts have `n_txns_below_threshold = 0`, fraud accounts have 3–7.

**Key insight**: the `declared_income_ratio` feature is the most discriminating. A merchant processing ₹28L in cash is plausible at ₹8L declared monthly income. The same deposits from someone declaring ₹40,000/month is almost certainly fraud.

**Training result**: F1=1.000 on synthetic data. Perfect scores are a symptom of the simulator generating structuring patterns with very consistent features. Will hold for demo since demo scenarios come from the same generator. Real-world F1 would be lower.

**Fallback**: calibrated rule fires if model not loaded: 3 suspicious txns → 0.60, 4–5 → 0.70, 6+ → 0.80.

---

### Dormancy Scorer

**Stage 1 — State machine**: account must be dormant (12+ months) AND current transaction > 10× historical average. Partial score 0.35–0.55 for accounts that are dormant but receive small amounts.

**Stage 2 — Logistic regression**: when state machine fires (base 0.75), adjusts based on dormancy duration, outbound speed, KYC update flag, income ratio, KYC tier.

**Training result**: F1=0.933, AUC=0.983. Strongest model. Recall 0.982 — catches 98.2% of dormant activation fraud.

---

### AnomaScore Aggregator

```
AnomaScore = 0.25 × layering
           + 0.30 × circular
           + 0.20 × structuring
           + 0.10 × dormancy
           + 0.15 × profile_mismatch
```

Circular gets highest weight (0.30) — deterministic algorithm, lowest false positive rate. Profile mismatch gets lowest (0.15) — requires trained autoencoder, contributes 0.0 until Rupali's module is ready.

Weights are runtime-configurable via PUT `/ml/weights` without redeployment.

Alert thresholds: standard ≥ 0.65, PEP tier ≥ 0.45.

Every detector call is in try/except — one failure never kills the pipeline.

---

## Part 4 — interfaces.py

The single integration file between Muskan and Rupali. Three functions:

```python
score_profile_mismatch(account_id) → float   # Rupali's LSTM autoencoder
get_rolling_features(account_id)   → dict    # Rupali's Redis sorted sets
get_explanation(alert_id, breakdown) → str   # Rupali's template engine
```

Each function tries to import Rupali's real implementation. If her module is not ready, a safe fallback runs: `score_profile_mismatch` returns 0.0, `get_rolling_features` returns a zero-value dict, `get_explanation` returns a generic paragraph built from the score breakdown. Muskan's pipeline never crashes waiting for Rupali.

---

## Part 5 — Training

**`training/train_classifiers.py`** trains all three classical models on the 100k dataset.

Feature engineering notes:
- Isolation Forest trains on **clean accounts only** — it learns what normal looks like, then flags deviations. Contamination parameter set to the actual fraud rate.
- XGBoost trains on **all cash accounts** — clean accounts have `n_txns_below_threshold = 0` which is itself a strong negative feature.
- Logistic Regression trains on **dormant accounts only** — restricting to the relevant population gives much cleaner signal.

Models saved to `core/models/`:
```
isolation_forest_layering.pkl    F1=0.025  AUC=0.853  (rule compensates)
xgboost_structuring.pkl          F1=1.000  AUC=1.000  (synthetic data)
logistic_dormancy.pkl            F1=0.933  AUC=0.983  ← production quality
```

---

## Part 6 — FastAPI Service

**`core/main.py`** — the HTTP interface for the ML layer.

**Endpoints**:

| Method | Path | Purpose |
|---|---|---|
| POST | `/ml/score` | Score a transaction — called by Ratnesh's transaction-service |
| POST | `/ml/explain` | Explanation for an alert — Rupali's router replaces this |
| GET | `/ml/health` | Models loaded, Neo4j status, uptime |
| GET | `/ml/model-info` | Model versions, weights, thresholds |
| PUT | `/ml/weights` | Update AnomaScore weights at runtime |
| POST | `/simulator/trigger` | Trigger a fraud scenario for demo |

Rupali's explainability router plugs in with one line at startup:
```python
app.include_router(explain_router)
```
If her module is not ready, the fallback `/ml/explain` endpoint runs using `interfaces.get_explanation()`.

---

## Part 7 — Kafka Consumer

**`core/kafka/consumer.py`** — consumes `scoring.queue`, runs the full pipeline per message, publishes `AlertEvent` to `alerts.generated`.

Per message:
1. Deserialise EnrichmentEvent from Ratnesh's transaction-service
2. Write TRANSFERRED_TO edge to Neo4j
3. Compute residency time (inbound→outbound speed)
4. Fetch recent transactions for structuring scorer
5. Run `compute_anoma_score()` — all 5 detectors
6. Update `anoma_score` on Account node in Neo4j (feeds D3 graph colours)
7. If alert triggered → publish AlertEvent to `alerts.generated`
8. On failure → dead-letter queue, continue to next message

Resilience: exponential backoff on Kafka connection loss (5s → 60s max). One failed message never blocks the queue. DLQ captures failures for manual review.

---

## Trained model metrics summary

| Model | Detector | F1 | AUC | Notes |
|---|---|---|---|---|
| Isolation Forest | Layering | 0.025 | 0.853 | Hard rule compensates. AUC shows ranking is good. |
| XGBoost | Structuring | 1.000 | 1.000 | Synthetic data. Will hold for demo. |
| Logistic Regression | Dormancy | 0.933 | 0.983 | Production quality. Recall 98.2%. |

---

## How to set up from scratch

```powershell
# Step 1 — start infrastructure (from repo root AnomaNet/)
docker-compose up -d
# Wait 60 seconds, verify: docker-compose ps

# Step 2 — Python environment (from AnomaNet/ml/)
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt

# Step 3 — generate training dataset (once, ~3-5 min)
python -m data_simulator.simulator --output data

# Step 4 — load dataset into Neo4j (once)
python -c "from core.graph.neo4j_client import bulk_load_from_simulator; bulk_load_from_simulator()"

# Step 5 — train all models (once, ~30 seconds)
python -m training.train_classifiers --no-mlflow

# Step 6 — start ML inference service
uvicorn core.main:app --reload --port 8000
```

---

## How to test everything

### Tests with no dependencies (run anytime)

**Circular detector — valid fraud cycle**
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
print('Detected:', r.cycle_detected, '| Score:', r.cycle_score, '| Path:', r.cycle_path)
"
```
Expected: `Detected: True`, score 0.70–0.95

---

**Circular detector — rejects bad cycle (amount variance too high)**
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
print('Detected:', r.cycle_detected, '| Score:', r.cycle_score)
"
```
Expected: `Detected: False`, score 0.0

---

**Layering scorer — fires on velocity**
```powershell
python -c "
from core.scoring.layering_scorer import score_layering_from_features
r = score_layering_from_features({
    'tx_count_1h': 8, 'total_amount_1h': 4500000,
    'unique_counterparties_24h': 7, 'cross_branch_ratio': 0.85,
    'residency_seconds': 420, 'is_off_hours': 1,
}, account_id='TEST')
print('Score:', r.layering_score, '| Rule fired:', r.rule_fired)
"
```
Expected: score ≥ 0.85, rule fired = True

---

**Layering scorer — passes clean account**
```powershell
python -c "
from core.scoring.layering_scorer import score_layering_from_features
r = score_layering_from_features({
    'tx_count_1h': 2, 'total_amount_1h': 45000,
    'unique_counterparties_24h': 2, 'cross_branch_ratio': 0.0,
    'residency_seconds': 9999, 'is_off_hours': 0,
}, account_id='CLEAN')
print('Score:', r.layering_score, '| Rule fired:', r.rule_fired)
"
```
Expected: score 0.0, rule fired = False

---

**Structuring scorer — catches threshold clustering**
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
print('Score:', r.structuring_score, '| Tier: INR', r.threshold_tier, '| Txns:', r.n_suspicious_txns)
"
```
Expected: score ≥ 0.65, tier = 1000000, txns = 4

---

**Structuring scorer — ignores non-cash transactions**
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
Expected: score 0.0

---

### Tests requiring docker-compose up

**Neo4j reachable**
```powershell
python -c "from core.graph.neo4j_client import health_check; print('Neo4j:', health_check())"
```
Expected: `Neo4j: True`

**ML service health**
```powershell
curl http://localhost:8000/ml/health
```
Expected: `{"status":"ok", "models_loaded":["isolation_forest_layering","xgboost_structuring","logistic_dormancy"], ...}`

**Score a structuring transaction via HTTP**
```powershell
curl -X POST http://localhost:8000/ml/score `
  -H "Content-Type: application/json" `
  -d "{\"transaction_id\":\"test-001\",\"account_id\":\"999888777666\",\"amount\":960000,\"channel\":\"CASH\",\"initiated_at\":\"2026-03-20T14:00:00+00:00\",\"kyc_risk_tier\":\"LOW\",\"declared_monthly_income\":40000,\"recent_transactions\":[{\"amount\":940000,\"channel\":\"CASH\",\"branch_id\":\"HDFC0001\",\"initiated_at\":\"2026-03-18T10:00:00+00:00\"},{\"amount\":975000,\"channel\":\"BRANCH\",\"branch_id\":\"ICIC0001\",\"initiated_at\":\"2026-03-19T11:00:00+00:00\"}]}"
```
Expected: `alert_triggered: true`, `detected_patterns: ["STRUCTURING"]`

---

### Tests requiring Neo4j loaded with simulator data

**Subgraph extraction**
```powershell
python -c "
import pandas as pd
from core.graph.neo4j_client import get_subgraph
tx = pd.read_parquet('data/transactions.parquet')
account_id = tx[tx.fraud_type == 'CIRCULAR'].iloc[0]['source_account_id']
G = get_subgraph(account_id, hops=3, hours=720)
print('Account:', account_id, '| Nodes:', G.number_of_nodes(), '| Edges:', G.number_of_edges())
"
```
Expected: nodes > 1, edges > 0

**Circular detector on real fraud account**
```powershell
python -c "
import pandas as pd
from core.graph.neo4j_client import get_subgraph
from core.scoring.circular_detector import score_circular_from_graph
tx = pd.read_parquet('data/transactions.parquet')
account_id = tx[tx.fraud_type == 'CIRCULAR'].iloc[0]['source_account_id']
G = get_subgraph(account_id, hops=7, hours=720)
r = score_circular_from_graph(G, account_id)
print('Detected:', r.cycle_detected, '| Score:', r.cycle_score)
"
```
Expected: `Detected: True`, score > 0.70

**Fire a scenario end-to-end**
```powershell
curl -X POST "http://localhost:8000/simulator/trigger?type=CIRCULAR"
# Watch uvicorn logs — should show CIRCULAR detected, AlertEvent published
```

---

## Directory structure

```
ml/
├── README.md
├── requirements.txt
├── interfaces.py                   ← Muskan creates, Rupali implements
├── data/                           ← gitignored — regenerate with simulator
│   ├── transactions.parquet        100k transactions
│   ├── accounts.parquet
│   ├── customers.parquet
│   ├── labels.parquet
│   ├── neo4j_nodes.parquet
│   └── neo4j_edges.parquet
│
├── data_simulator/
│   ├── models.py                   shared types + helpers (no circular import risk)
│   ├── simulator.py                orchestrator — generates 100k transactions
│   └── scenarios/
│       ├── layering.py
│       ├── circular.py
│       ├── structuring.py
│       ├── dormant_activation.py
│       └── profile_mismatch_gen.py
│
├── core/
│   ├── main.py                     FastAPI app — entry point, all HTTP endpoints
│   ├── graph/
│   │   └── neo4j_client.py         all Neo4j interactions for ML layer
│   ├── scoring/
│   │   ├── circular_detector.py    Johnson's Algorithm cycle detection
│   │   ├── layering_scorer.py      Isolation Forest + hard velocity rule
│   │   ├── structuring_scorer.py   XGBoost threshold clustering classifier
│   │   ├── dormancy_scorer.py      state machine + logistic regression
│   │   └── anoma_score.py          weighted composite aggregator
│   ├── kafka/
│   │   └── consumer.py             scoring.queue → pipeline → alerts.generated
│   ├── gnn/                        🔲 not built yet — GraphSAGE encoder
│   └── models/                     .pkl files written here after training
│       ├── isolation_forest_layering.pkl
│       ├── xgboost_structuring.pkl
│       └── logistic_dormancy.pkl
│
├── training/
│   └── train_classifiers.py        trains all 3 models, saves to core/models/
│
├── modules/                        Rupali's modules — do not touch
│   ├── profile_mismatch/
│   └── explainability/
│
└── shared/                         Rupali's Redis feature store — do not touch
    └── feature_store/
```

---

## What is not built yet

| File | Purpose | Priority |
|---|---|---|
| `core/gnn/graphsage_encoder.py` | 3-layer GraphSAGE, 128-dim account embeddings capturing structural fraud proximity | P1 — impressive to judges, not blocking demo |
| `training/train_gnn.py` | Semi-supervised training loop with MLflow, F1 target > 0.80 | P1 — needed after encoder |

The GNN is the most technically ambitious component remaining. It encodes each account's position in the fraud network into a 128-dimensional vector — meaning an account that looks clean on its own but sits two hops from five known fraud accounts will have a high-risk embedding. This is what separates AnomaNet from all other entries.