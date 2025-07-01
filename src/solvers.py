import numpy as np
from scipy.sparse.linalg import svds
from tqdm import trange

def atoms_equal(A, B, tol=1e-8):
    return np.allclose(A, B, atol=tol)

class FrankWolfe:
    def __init__(
        self,
        objective,
        lmo_fn,
        tau: float = 1.0,
        max_iter: int = 50,
        tol: float = 1e-4,
        step_method: str = 'vanilla',
        fixed_gamma: float = 0.1,
        func_tol: float = 1e-8,
        verbose: bool = False
    ):
        self.obj = objective
        self.lmo = lmo_fn
        self.tau = tau
        self.max_iter = max_iter
        self.tol = tol
        self.step_method = step_method
        self.fixed_gamma = fixed_gamma
        self.func_tol = func_tol
        self.verbose = verbose
        self.history = []

    def _dual_gap(self, X, grad, S):
        raw = -np.sum((X - S) * grad)
        return max(raw, 1e-16)

    def _choose_step(self, X, grad, S, d, t):
        """
        Return step-size alpha in [0,1] for the search direction d = S - X.

        Available self.step_method values:
            - 'fixed'        : use constant self.fixed_gamma
            - 'vanilla'      : alpha_k = 2/(t+2)
            - 'analytic'     : alpha_k = min(max(-<grad,d>/<d,d>_mask, 0), 1)
            - 'line_search'  : brute-force on a small grid
            - 'armijo'       : backtracking Armijo rule
        """
        if self.step_method == 'fixed':
            return float(self.fixed_gamma)

        if self.step_method == 'vanilla':
            return 2.0 / (t + 2)

        if self.step_method == 'analytic':
            num = float(np.sum(grad * d))
            den = float(np.sum((d * self.obj.mask) ** 2))
            if den <= 0:
                return 2.0 / (t + 2)
            alpha = -num / den
            return float(np.clip(alpha, 0.0, 1.0))

        if self.step_method == 'line_search':
            alphas = np.linspace(0.0, 1.0, 20)
            vals = [self.obj.value(X + a * d) for a in alphas]
            return float(alphas[np.argmin(vals)])

        if self.step_method == 'armijo':
            delta, beta = 0.5, 0.1
            fk = self.obj.value(X)
            gd = float(np.sum(grad * d))
            alpha = 1.0
            while alpha > 1e-8:
                if self.obj.value(X + alpha * d) <= fk + beta * alpha * gd:
                    return alpha
                alpha *= delta
            return 0.0

        return 2.0 / (t + 2)

    def run(self, X0=None):
        X = np.zeros_like(self.obj.M_obs) if X0 is None else X0.copy()
        prev_obj = self.obj.value(X)

        for t in trange(self.max_iter, desc='Frank-Wolfe'):
            grad = self.obj.gradient(X)
            S = self.lmo(grad, self.tau)
            d = S - X
            gap = self._dual_gap(X, grad, S)
            obj_val = self.obj.value(X)
            self.history.append((t, gap, obj_val))

            if self.verbose:
                print(f"[{t}] gap: {gap:.2e}  f(X): {obj_val:.6f}")

            if gap < self.tol or (t > 0 and abs(obj_val - prev_obj) < self.func_tol):
                break

            gamma = self._choose_step(X, grad, S, d, t)
            X += gamma * d
            prev_obj = obj_val

        return X


class PairwiseFrankWolfe(FrankWolfe):
    def run(self, X0=None):
        X = np.zeros_like(self.obj.M_obs) if X0 is None else X0.copy()
        atoms, weights = [], []

        for t in trange(self.max_iter, desc='Pairwise-FW'):
            grad = self.obj.gradient(X)
            S = self.lmo(grad, self.tau)

            found = False
            for i, A in enumerate(atoms):
                if atoms_equal(S, A):
                    idx_S = i
                    found = True
                    break
            if not found:
                atoms.append(S)
                weights.append(0.0)
                idx_S = len(atoms) - 1

            scores = [np.sum(grad * A) for A in atoms]
            idx_max = int(np.argmax(scores))
            V = atoms[idx_max]
            alpha_max = weights[idx_max]

            d = S - V
            gap = self._dual_gap(X, grad, S)
            obj_val = self.obj.value(X)
            self.history.append((t, gap, obj_val))

            if self.verbose:
                print(f"[{t}] gap: {gap:.2e}  f(X): {obj_val:.6f}")

            if gap < self.tol or (t > 0 and abs(obj_val - self.history[-2][2]) < self.func_tol):
                break

            gamma = min(self._choose_step(X, grad, S, d, t), alpha_max)
            X += gamma * d
            weights[idx_S] += gamma
            weights[idx_max] -= gamma

            nz = [(a, w) for a, w in zip(atoms, weights) if w > 1e-12]
            if nz:
                atoms, weights = map(list, zip(*nz))
            else:
                atoms, weights = [], []

        return X
