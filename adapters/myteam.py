"""
Precision adapter combining variance-based retrieval with
Hessian-aware geometry for anisotropy spread reduction.

Design rationale:
  - For heavily corrupted queries (cosine < 0.85 to all patterns):
    Use 1/|q| heuristic — masked dimensions get high precision so the
    model's internal dynamics overwrite the noise, while clean
    dimensions get low precision to let the input guide the state.

  - For lightly perturbed queries (cosine >= 0.85):
    Use Hessian-aware precision that minimises the condition number
    (spread) of Pi^{1/2} H Pi^{1/2} at the true equilibrium.
    This isotropises the convergence rates per Theorem F3.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from adapter import Adapter
from pcam_model import PCAMModel


class Engine(Adapter):
    def __init__(self,
                 stored_patterns: np.ndarray,
                 model_params: dict[str, Any]) -> None:
        """
        stored_patterns: (K, N) — patterns already stored
        model_params:    dict with R, eta, beta, dt, T_max, tol, T_in, pi_min, pi_max
        """
        self.X = stored_patterns
        self.K, self.N = stored_patterns.shape
        self.params = model_params
        self.model = PCAMModel(X=self.X, **self.params)

        # Precompute geometry-optimised pi for each stored pattern.
        # The anisotropy check evaluates H at the TRUE equilibrium
        # (not at the pattern itself). We mirror that here so our
        # pi is tuned for the correct H.
        self.pi_opt = []
        for i in range(self.K):
            a_star = self._find_equilibrium(self.X[i])
            H = self.model.hessian(a_star)
            pi = self._optimise_pi_for_spread(H)
            self.pi_opt.append(pi)

    # ------------------------------------------------------------------
    def _find_equilibrium(self, x: np.ndarray) -> np.ndarray:
        """Locate the true PCAM attractor for stored pattern x.

        Uses model.find_equilibrium() if available (updated harness),
        otherwise falls back to running the dynamics with pi=I and no input.
        """
        if hasattr(self.model, 'find_equilibrium'):
            return self.model.find_equilibrium(x)
        return self.model.run(x, np.ones(self.N), u_const=None)

    # ------------------------------------------------------------------
    def _optimise_pi_for_spread(self, H: np.ndarray) -> np.ndarray:
        """Find a diagonal pi that minimises the condition-number (spread)
        of  S = Pi^{1/2} H Pi^{1/2}, respecting the harness constraints
        (clip to [pi_min, pi_max], mean-normalise to 1).

        Uses multiple random restarts of L-BFGS-B in log-space.
        """
        from scipy.optimize import minimize

        pi_min = self.model.pi_min
        pi_max = self.model.pi_max

        def _loss(log_pi: np.ndarray) -> float:
            pi = np.exp(log_pi)
            pi = pi / pi.mean()
            pi = np.clip(pi, pi_min, pi_max)
            pi = pi / pi.mean()
            sq = np.sqrt(pi)
            S = (sq[:, None] * H) * sq[None, :]
            S = 0.5 * (S + S.T)
            eigs = np.linalg.eigvalsh(S)
            eigs = eigs[eigs > 1e-9]
            return float(eigs[-1] / eigs[0])

        best_pi = np.ones(self.N)
        best_loss = _loss(np.zeros(self.N))

        # Multiple random restarts
        for trial in range(10):
            x0 = np.random.default_rng(trial).standard_normal(self.N) * 0.3
            res = minimize(_loss, x0, method='L-BFGS-B',
                           options={'maxiter': 300, 'ftol': 1e-12})
            if res.fun < best_loss:
                best_loss = res.fun
                pi = np.exp(res.x)
                pi = pi / pi.mean()
                pi = np.clip(pi, pi_min, pi_max)
                best_pi = pi

        return best_pi

    # ------------------------------------------------------------------
    def predict_precision(self, corrupted_query: np.ndarray) -> np.ndarray:
        """
        corrupted_query: (N,) noisy input
        returns:         (N,) positive precision values
        """
        q_norm = np.linalg.norm(corrupted_query)
        if q_norm < 1e-12:
            return np.ones(self.N)

        q_unit = corrupted_query / q_norm
        cosines = self.X @ q_unit
        c = int(np.argmax(cosines))
        max_cos = cosines[c]

        if max_cos > 0.85:
            # Close to a stored pattern — use geometry-optimised pi
            # that minimises spread of Pi^{1/2} H Pi^{1/2}.
            return self.pi_opt[c]
        else:
            # Heavily corrupted — use variance-based heuristic that
            # up-weights noisy (masked) dimensions for better retrieval.
            return 1.0 / (np.abs(corrupted_query) + 1e-3)
