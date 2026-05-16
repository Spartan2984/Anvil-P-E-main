"""Generate PCA-MNIST-like mock patterns and run a quick evaluation.

Creates a low-rank, class-structured pattern matrix with a decaying
singular-value spectrum to mimic PCA'd MNIST-like structure. Saves
the generated dataset to `mock_data/pca_mnist_like.npz` and prints a
brief retrieval + anisotropy comparison between the dummy and
`adapters.myteam.Engine` agents.
"""
from __future__ import annotations

import os
import sys
import numpy as np

# Ensure the project root is on sys.path so `from data import ...` works
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data import make_test_queries
from pcam_model import PCAMModel, build_default_R
from metrics import (retrieval_accuracy, direct_classify_accuracy,
                     anisotropy_reductions, summarise_anisotropy)


def make_pca_like_patterns(K: int, N: int, seed: int,
                           n_clusters: int = 10,
                           latent_dim: int | None = None,
                           sv_power: float = 1.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    latent_dim = latent_dim or min(32, N, K)

    # Random orthonormal basis (N x L)
    A = rng.standard_normal((N, latent_dim))
    Q, _ = np.linalg.qr(A)

    # Power-law singular values to mimic PCA spectrum (decaying)
    idx = np.arange(1, latent_dim + 1)
    sv = 1.0 / (idx ** sv_power)

    # Cluster centres in latent space
    n_clusters = max(2, min(n_clusters, K))
    centers = rng.standard_normal((n_clusters, latent_dim)) * 1.0
    centers = centers / (np.linalg.norm(centers, axis=1, keepdims=True) + 1e-12)

    Z = np.empty((K, latent_dim), dtype=np.float64)
    for k in range(K):
        c = k % n_clusters
        z = centers[c] + rng.standard_normal(latent_dim) * 0.3
        Z[k] = z

    # Project to high-dim with decaying spectrum
    X = (Q * sv[None, :]) @ Z.T
    X = X.T
    # Normalise patterns to unit norm
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    X = X / norms
    return X


def main():
    out_dir = "mock_data"
    os.makedirs(out_dir, exist_ok=True)

    K = 16
    N = 64
    seed = 42

    X = make_pca_like_patterns(K=K, N=N, seed=seed,
                               n_clusters=4, latent_dim=16, sv_power=0.8)

    # Build model
    R = build_default_R(N=N, seed=seed)
    model = PCAMModel(X, R)
    params = {
        "R": model.R,
        "eta": model.eta,
        "beta": model.beta,
        "dt": model.dt,
        "T_max": model.T_max,
        "tol": model.tol,
        "T_in": model.T_in,
        "pi_min": model.pi_min,
        "pi_max": model.pi_max,
    }

    # Build agents
    from adapters.dummy import DummyAgent
    from adapters.myteam import Engine as MyAgent

    dummy = DummyAgent(X, params)
    myagent = MyAgent(X, params)

    # Queries
    noise_levels = [0.6, 0.75, 0.85]
    queries, truths, levels = make_test_queries(X, noise_levels, n_per_level=80, seed=seed)

    # Evaluate retrieval
    direct_acc = direct_classify_accuracy(model, queries, truths)
    dummy_acc = retrieval_accuracy(model, dummy, queries, truths)
    my_acc = retrieval_accuracy(model, myagent, queries, truths)

    print("Retrieval: direct={:.3f}, dummy={:.3f}, myagent={:.3f}".format(direct_acc, dummy_acc, my_acc))

    # Anisotropy (sample a few patterns)
    inds = list(range(min(8, K)))
    pairs_dummy = anisotropy_reductions(model, dummy, inds, seed=seed)
    pairs_my = anisotropy_reductions(model, myagent, inds, seed=seed)
    summary_dummy = summarise_anisotropy(pairs_dummy)
    summary_my = summarise_anisotropy(pairs_my)

    print("Anisotropy baseline_spread={:.2f}, agent_spread={:.2f}, reduction={:.2f}x".format(
        summary_dummy["baseline_spread"], summary_dummy["agent_spread"], summary_dummy["reduction"]))
    print("MyAgent    baseline_spread={:.2f}, agent_spread={:.2f}, reduction={:.2f}x".format(
        summary_my["baseline_spread"], summary_my["agent_spread"], summary_my["reduction"]))

    # Save mock dataset
    np.savez_compressed(
        os.path.join(out_dir, "pca_mnist_like.npz"),
        X=X, queries=queries, truths=truths, levels=levels
    )

    print(f"Saved mock dataset to {out_dir}/pca_mnist_like.npz")


if __name__ == "__main__":
    import traceback
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}")
        traceback.print_exc()
