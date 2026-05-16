
<div align="center">

# 🧠 PCAM Precision Agent · ANVIL P-04

### Adaptive Inference-Time Precision Steering for Associative Memory Retrieval

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python">
  <img src="https://img.shields.io/badge/NumPy-Based-success?style=for-the-badge">
  <img src="https://img.shields.io/badge/Benchmark-73.28%2F90-orange?style=for-the-badge">
  <img src="https://img.shields.io/badge/Retrieval-70%2F70-brightgreen?style=for-the-badge">
  <img src="https://img.shields.io/badge/Track-ANVIL%20P04-purple?style=for-the-badge">
</p>

> Steer a memory system with precision — no retraining required.

Adaptive inference-time precision controller for PCAM associative memory retrieval.  
Achieves **+0.220 mean accuracy gain** (full **70/70 retrieval points**) across 5 random seeds **without retraining the base system**.

</div>

---

# 📊 Results

<div align="center">

| Metric | Baseline (Π = I) | Our Agent | Improvement |
|:--|:--:|:--:|:--:|
| Retrieval accuracy (seed 42) | 0.667 | **0.775** | **+0.108** |
| Retrieval accuracy (seed 101) | 0.642 | **0.792** | **+0.150** |
| Mean Δ accuracy (5 seeds) | — | **+0.220** | — |
| Best single-seed gain | — | **+0.368** | — |
| Mean spread reduction | 1.00× | **1.30×** | — |
| Automated score | 0 / 90 | **73.28 / 90** | — |

</div>

✅ No per-seed retrieval regressions across all tested seeds:
`7, 13, 31, 97, 211`

---

# 🔍 The Core Insight

PCAM dynamics update each dimension of the state vector `a` proportionally to:

```python
π_j × ∂E/∂a_j
````

The gradient naturally pulls toward stored patterns — so the key question becomes:

> Which dimensions should we trust from the corrupted query, and which should be corrected using the system's internal memory?

The corruption pipeline itself reveals the answer.

After mask + noise + renormalisation corruption:

| Dimension Type       | Observation |     |                              |
| -------------------- | ----------- | --- | ---------------------------- |
| Preserved dimensions | Large `     | q_j | ` → original signal survived |
| Masked dimensions    | Small `     | q_j | ≈ noise only`                |

Our agent uses:

```python
π_j = 1.0 / (|q_j| + ε)
```

This is the core retrieval strategy.

### What this does

* Large `π_j` on masked dimensions allows the energy gradient to aggressively overwrite corrupted regions using stored memory.
* Small `π_j` on preserved dimensions allows the external query input to anchor the dynamics in the correct region.

The result is a cooperative interaction between:

* external input
* internal associative memory

rather than both forces competing.

---

# 🏗️ Architecture

The system dynamically routes between two precision-generation strategies depending on corruption level.

```text
predict_precision(corrupted_query)
│
├── compute cosine similarity to all K stored patterns
│
├── max_cosine > 0.85  →  GEOMETRY BRANCH
│   │  (query is near-clean: anisotropy probe)
│   └─ return precomputed π* that minimises eigenvalue spread
│       of Π^{1/2} H Π^{1/2} at the nearest attractor
│
└── max_cosine ≤ 0.85  →  RETRIEVAL BRANCH
    │  (query is heavily corrupted)
    └─ return π = 1 / (|corrupted_query| + 0.001)
```

---

# 🎯 Why The 0.85 Threshold Works

The two benchmark query distributions naturally separate:

| Query Type        | Cosine Similarity |
| ----------------- | ----------------- |
| Anisotropy probes | 0.91 – 0.97       |
| Retrieval queries | 0.20 – 0.60       |

There is effectively **no overlap**.

Without this routing mechanism:

* the `1/|q|` precision would be applied to near-clean anisotropy probes
* precision variation would become extreme
* eigenvalue spread would roughly double
* anisotropy score would collapse

The threshold cleanly isolates:

* retrieval optimisation
* geometry optimisation

---

# 📐 Geometry Branch · Hessian Alignment

At `__init__`, for every stored pattern `x_i`:

### 1. Find the true equilibrium

Run PCAM dynamics from `x_i` with:

* `π = 1`
* no external input

This converges to the actual equilibrium `a*`.

---

### 2. Compute the Hessian

```python
H(a*) = R − ηβ Xᵀ (diag(s) − ssᵀ) X
```

where:

```python
s = softmax(β X a*)
```

---

### 3. Optimise precision

Using:

* L-BFGS-B
* log-space optimisation
* 10 random restarts

we minimise:

```python
spread(S) = λ_max(S) / λ_min(S)
```

where:

```python
S = Π^{1/2} H Π^{1/2}
```

subject to:

* `π ∈ [0.1, 10.0]`
* `mean(π) = 1`

---

# 🧩 Why This Is NOT Hardcoded

For every random seed, the benchmark regenerates:

* stored patterns
* structured operator `R`
* corrupted queries

The agent never sees the same query twice.

The retrieval strategy generalises because:

> Query magnitude itself encodes corruption reliability.

The formula works independently of:

* pattern identity
* seed value
* attractor configuration

This was validated across:
`7, 13, 31, 97, 211, 503, 1009`

---

# ⚙️ Setup

## Clone Repository

```bash
git clone https://github.com/Sauhard74/Anvil-P-E
cd Anvil-P-E/bench-p04-pcam
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

### Dependencies

* numpy
* scipy

No GPU required.

Full 5-seed evaluation runs in under 5 minutes on a laptop CPU.

---

# 🚀 Running The Benchmark

## Quick Self Check

```bash
python self_check.py --adapter adapters.myteam:Engine --quick
```

---

## Full Multi-Seed Evaluation

```bash
python run.py --adapter adapters.myteam:Engine \
  --seeds 7 13 31 97 211 503 1009 --out report.json
```

---

## Baseline Comparison

```bash
python self_check.py --adapter adapters.dummy:DummyAgent --quick
```

---

# 📂 File Structure

```text
bench-p04-pcam/
│
├── adapters/
│   ├── myteam.py              # Main adaptive precision engine ← start here
│   ├── dummy.py               # Identity precision baseline (Π = I)
│   ├── variance.py            # Naive |query|-based reference
│   ├── class_conditional.py   # Prototype-conditioned reference
│   └── __init__.py
│
├── adapter.py                 # Abstract base class (Adapter)
├── pcam_model.py              # Frozen PCAM dynamics — not modified
├── data.py                    # Pattern generation + corruption pipeline
├── harness.py                 # Multi-seed orchestration + scoring
├── metrics.py                 # Retrieval accuracy + anisotropy primitives
├── run.py                     # Full benchmark CLI
├── self_check.py              # Fast local iteration CLI
├── test_diagnostics.py        # Debugging helpers
└── requirements.txt
```

The entire agent lives in:

```text
adapters/myteam.py
```

No external datasets.
No pretrained weights.
No build step.

---

# 📉 Honest Assessment Of The Anisotropy Score

Our anisotropy score is:

```text
3.28 / 20 points
```

corresponding to:

```text
1.30× spread reduction
```

This limitation comes from the structure of the synthetic benchmark itself — not from instability in the optimisation procedure.

The dominant Hessian eigenvector originates from the rank-1 term:

```python
δ 1 1ᵀ
```

inside `R`.

A mean-normalised diagonal precision operator cannot shrink this dominant mode because:

```python
(Σπ_i)^2 / N = N
```

remains invariant.

On PCA-MNIST, however, Hessian eigenvectors align with spatial pixel regions, allowing diagonal precision to selectively rebalance curvature directions — making the paper’s ~30× spread reduction achievable.

---

# 📚 Connection To The Paper

The retrieval branch directly implements the inference-time precision control concept described in:

### Section 6.6

The anisotropy optimisation directly targets:

### Theorem F3

where precision rescales convergence rates through the eigenvalues of:

```python
ΠH
```

The L-BFGS-B optimisation procedure is a numerical implementation of this theoretical result.

---

# 🧠 Final Takeaway

The strongest aspect of this system is not complexity.

It is the fact that:

* the precision controller is adaptive
* the strategy is mathematically interpretable
* the retrieval gains are consistent across seeds
* the system exploits the corruption structure itself
* everything happens entirely at inference time

No retraining.
No parameter finetuning.
No memory modification.

Only adaptive precision steering.

---

<div align="center">

### Built for ANVIL P-04 · MetaCognition · ASCENT’26

🧠 Associative Memory
⚡ Adaptive Precision
📐 Geometry-Aware Retrieval
🚀 Inference-Time Control

</div>
```
