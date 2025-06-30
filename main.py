#!/usr/bin/env python3
# matrix_completion_fw.py

import numpy as np
from scipy.sparse.linalg import svds
from tqdm import trange


# === 1) Dataset & Utilities ===

def load_dataset(name="synthetic", test_fraction=0.2, seed=0):
    """
    Load or generate a matrix completion dataset.
    Returns M_obs, mask_train, M_true.
    """
    rng = np.random.RandomState(seed)
    if name == "synthetic":
        n, m, r = 100, 80, 5
        U = rng.randn(n, r)
        V = rng.randn(m, r)
        M_true = U @ V.T
    else:
        data = np.load(name)
        M_true = data["M"]

    # train/test split
    mask_train = rng.rand(*M_true.shape) < (1 - test_fraction)
    M_obs = np.zeros_like(M_true)
    M_obs[mask_train] = M_true[mask_train]

    return M_obs, mask_train, M_true


def evaluate(M_true, X_pred, mask):
    """
    Compute RMSE on entries where mask is True.
    """
    diff = (X_pred - M_true)[mask]
    if diff.size == 0:
        return 0.0
    return np.sqrt(np.mean(diff ** 2))


# === 2) Objective Function ===

class MatrixCompletionObjective:
    def __init__(self, M_obs: np.ndarray, mask: np.ndarray):
        """
        Half squared error on observed entries.

        M_obs : observed entries (zeros elsewhere)
        mask  : boolean mask of observed entries
        """
        self.M_obs = M_obs
        self.mask = mask

    def value(self, X: np.ndarray) -> float:
        diff = (X - self.M_obs) * self.mask
        return 0.5 * np.sum(diff ** 2)

    def gradient(self, X: np.ndarray) -> np.ndarray:
        grad = np.zeros_like(X)
        grad[self.mask] = (X - self.M_obs)[self.mask]
        return grad


# === 3) Nuclear-Norm LMO ===

def nuclear_norm_lmo(grad: np.ndarray, tau: float) -> np.ndarray:
    """
    LMO over nuclear-norm ball: -tau * u1 v1^T
    """
    u, s, vt = svds(-grad, k=1)
    return -tau * np.outer(u[:, 0], vt[0, :])


# === 4) Frank–Wolfe Solver ===

class FrankWolfe:
    def __init__(
        self,
        objective,
        lmo_fn,
        tau: float = 1.0,
        max_iter: int = 200,
        tol: float = 1e-4,
        step_method: str = 'line_search',
        fixed_gamma: float = 0.1
    ):
        """
        Frank–Wolfe solver for convex objectives over nuclear-norm ball.
        """
        self.obj = objective
        self.lmo = lmo_fn
        self.tau = tau
        self.max_iter = max_iter
        self.tol = tol
        self.step_method = step_method
        self.fixed_gamma = fixed_gamma
        self.history = []

    def _dual_gap(self, X, grad, S):
        return np.sum((X - S) * grad)

    def _choose_step(self, X, grad, S, d, t):
        if self.step_method == 'fixed':
            return self.fixed_gamma
        if self.step_method == 'vanilla':  # classical 2/(t+2)
            return 2.0 / (t + 2)
        # line_search on grid:
        gammas = np.linspace(0, 1, 20)
        values = [self.obj.value(X + g * d) for g in gammas]
        return gammas[np.argmin(values)]

    def run(self, X0=None):
        X = np.zeros_like(self.obj.M_obs) if X0 is None else X0.copy()
        for t in trange(self.max_iter, desc='Frank-Wolfe'):
            grad = self.obj.gradient(X)
            S = self.lmo(grad, self.tau)
            d = S - X
            gap = self._dual_gap(X, grad, S)
            self.history.append((t, gap, self.obj.value(X)))
            if gap < self.tol:
                break
            gamma = self._choose_step(X, grad, S, d, t)
            X += gamma * d
        return X


# === 5) Main Execution ===

def main():
    # Settings
    DATASET       = "synthetic"    # or path to .npz
    TEST_FRAC     = 0.2
    SEED          = 123
    TAU           = 10.0
    MAX_ITER      = 200
    TOL           = 1e-4
    STEP_METHOD   = "line_search"  # 'line_search', 'fixed', or 'vanilla'
    FIXED_GAMMA   = 0.1

    # 1) Load data
    M_obs, mask_train, M_true = load_dataset(DATASET, TEST_FRAC, SEED)

    # 2) Build objective
    obj = MatrixCompletionObjective(M_obs, mask_train)

    # 3) Configure solver
    solver = FrankWolfe(
        objective=obj,
        lmo_fn=nuclear_norm_lmo,
        tau=TAU,
        max_iter=MAX_ITER,
        tol=TOL,
        step_method=STEP_METHOD,
        fixed_gamma=FIXED_GAMMA
    )

    # 4) Run Frank–Wolfe
    X_pred = solver.run()

    # 5) Evaluate
    rmse_train = np.sqrt(2 * obj.value(X_pred) / mask_train.sum())
    rmse_test  = evaluate(M_true, X_pred, ~mask_train)
    print(f"Train RMSE: {rmse_train:.4f}")
    print(f" Test RMSE: {rmse_test:.4f}")


if __name__ == "__main__":
    main()
