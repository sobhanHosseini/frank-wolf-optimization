import numpy as np
from scipy.sparse.linalg import svds
from tqdm import trange
import time

# ----------------------------------------------------------------
# Problem definition and LMO
# ----------------------------------------------------------------
class MatrixCompletionObjective:
    def __init__(self, M_obs: np.ndarray, mask: np.ndarray):
        """
        Half squared error on observed entries.
        M_obs : observed entries (zeros elsewhere)
        mask  : boolean mask of observed entries
        """
        self.M_obs = M_obs
        self.mask  = mask

    def value(self, X: np.ndarray) -> float:
        diff = (X - self.M_obs)[self.mask]
        return 0.5 * float(np.dot(diff, diff))

    def gradient(self, X: np.ndarray) -> np.ndarray:
        grad = np.zeros_like(X)
        grad[self.mask] = (X - self.M_obs)[self.mask]
        return grad

# ----------------------------------------------------------------
# Linear Minimization Oracle (nuclear-norm ball)
# ----------------------------------------------------------------
def nuclear_norm_lmo(grad: np.ndarray, tau: float):
    """
    Returns S = -tau * u1 v1^T solving min ⟨grad, Z⟩ s.t. ||Z||_* ≤ tau.
    Note: negative ensures descent direction.
    """
    try:
        u, s, vt = svds(grad, k=1, tol=1e-3, maxiter=200)
        if s[0] < 0:
            raise RuntimeError("Non-positive singular value")
    except Exception:
        U, S, VT = np.linalg.svd(grad, full_matrices=False)
        u, vt    = U[:, :1], VT[:1, :]
    return Atom(u, vt, tau)

# ----------------------------------------------------------------
# Rank-1 Atom container
# ----------------------------------------------------------------
class Atom:
    def __init__(self, u: np.ndarray, v: np.ndarray, tau: float):
        self.u   = u.reshape(-1,1)
        self.v   = v.reshape(1,-1)
        self.tau = tau

    def to_matrix(self) -> np.ndarray:
        # Return -tau * u v^T to align with descent direction
        return -self.tau * (self.u @ self.v)

    def __eq__(self, other):
        return (
            isinstance(other, Atom) and
            np.allclose(self.u, other.u) and
            np.allclose(self.v, other.v)
        )

    def __repr__(self):
        return f"Atom(tau={self.tau}, u_shape={self.u.shape}, v_shape={self.v.shape})"

# ----------------------------------------------------------------
# Frank–Wolfe solver with logging, early RMSE stopping, and memory-efficient snapshots
# ----------------------------------------------------------------
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
        tol_obj: float = 1e-6,
        patience: int = 10,
        snapshot_dtype: type = np.float32
    ):
        print('test ...')
        # problem
        self.obj             = objective
        self.lmo             = lmo_fn
        self.tau             = tau
        self.max_iter        = max_iter
        self.tol             = tol
        self.step_method     = step_method
        self.fixed_gamma     = fixed_gamma
        self.tol_obj         = tol_obj
        self.patience        = patience
        self.snapshot_dtype  = snapshot_dtype  # for reduced memory
        # logging and histories
        self.history         = []  # (iter, gap, obj)
        self.times           = []  # wall-clock times
        self.snapshots       = []  # X iterates (float32)
        self.step_history    = []  # gamma iterates

    def _dual_gap(self, X, grad, S):
        raw = -np.sum((S - X) * grad)
        return float(max(raw, 0.0))

    def _choose_step(self, X, grad, S, d, t):
        if self.step_method == 'fixed':
            return float(self.fixed_gamma)
        if self.step_method == 'vanilla':
            return 2.0 / (t + 2)
        if self.step_method == 'analytic':
            num = float(np.sum(grad * d))
            den = float(np.sum((d * self.obj.mask)**2))
            if den <= 0:
                return 2.0 / (t + 2)
            return float(np.clip(-num/den, 0.0, 1.0))
        if self.step_method == 'line_search':
            alphas = np.linspace(0.0, 1.0, 20)
            vals   = [self.obj.value(X + a*d) for a in alphas]
            return float(alphas[np.argmin(vals)])
        if self.step_method == 'armijo':
            delta, beta = 0.5, 0.1
            fk = self.obj.value(X)
            gd = float(np.sum(grad * d))
            alpha = 1.0
            while alpha > 1e-8:
                if self.obj.value(X + alpha*d) <= fk + beta*alpha*gd:
                    return alpha
                alpha *= delta
            return 0.0
        return 2.0/(t+2)

    def run(self, X0=None):
        X = np.zeros_like(self.obj.M_obs) if X0 is None else X0.copy()
        # initial snapshot
        self.snapshots.append(X.astype(self.snapshot_dtype, copy=False).copy())
        t0 = time.perf_counter()
        self.times.append(0.0)

        best_obj, no_imp = float('inf'), 0
        last_rmse, rmse_no_imp = float('inf'), 0
        mask_count = float(self.obj.mask.sum())

        for t in trange(self.max_iter, desc='Frank-Wolfe'):
            grad    = self.obj.gradient(X)
            atom    = self.lmo(grad, self.tau)
            S       = atom.to_matrix()
            d       = S - X
            gap     = self._dual_gap(X, grad, S)
            obj_val = self.obj.value(X)
            # compute train RMSE
            rmse = np.sqrt(2 * obj_val / mask_count)

            self.history.append((t, gap, obj_val))
            if gap < self.tol:
                break
            # objective-based patience
            if obj_val < best_obj - self.tol_obj:
                best_obj, no_imp = obj_val, 0
            else:
                no_imp += 1
                if no_imp >= self.patience:
                    break
            # RMSE-based early stop
            if abs(last_rmse - rmse) < 1e-4:
                rmse_no_imp += 1
                if rmse_no_imp >= self.patience:
                    break
            else:
                last_rmse, rmse_no_imp = rmse, 0

            gamma = self._choose_step(X, grad, S, d, t)
            X    += gamma * d
            self.step_history.append(gamma)
            # reduced-memory snapshot
            self.snapshots.append(X.astype(self.snapshot_dtype, copy=False).copy())
            self.times.append(time.perf_counter() - t0)
        return X

# ----------------------------------------------------------------
# Pairwise Frank–Wolfe with optimized away selection, early RMSE stopping, and memory-efficient snapshots
# ----------------------------------------------------------------
class PairwiseFrankWolfe(FrankWolfe):
    def run(self, X0=None):
        X = np.zeros_like(self.obj.M_obs) if X0 is None else X0.copy()
        self.snapshots   = [X.astype(self.snapshot_dtype, copy=False).copy()]
        t0 = time.perf_counter()
        self.times       = [0.0]
        self.history     = []
        self.step_history = []
        self.weights_history = []

        atoms, weights = [], []
        best_obj, no_imp = float('inf'), 0
        last_rmse, rmse_no_imp = float('inf'), 0
        mask_count = float(self.obj.mask.sum())

        for t in trange(self.max_iter, desc='Pairwise-FW'):
            grad = self.obj.gradient(X)
            atom = self.lmo(grad, self.tau)
            if atom not in atoms:
                atoms.append(atom)
                weights.append(0.0)
            idx_S = atoms.index(atom)
            # fast bilinear scoring
            scores = [-a.tau * float(a.u.T @ grad @ a.v.T) for a in atoms]
            idx_away = int(np.argmax(scores))
            alpha_max = weights[idx_away]

            S = atom.to_matrix()
            if alpha_max <= 0:
                d = S - X
                gamma = self._choose_step(X, grad, S, d, t)
                weights[idx_S] += gamma
            else:
                V = atoms[idx_away].to_matrix()
                d = S - V
                gamma = min(self._choose_step(X, grad, S, d, t), alpha_max)
                weights[idx_S]   += gamma
                weights[idx_away] -= gamma

            X += gamma * d
            gap     = self._dual_gap(X, grad, S)
            obj_val = self.obj.value(X)
            # compute train RMSE
            rmse = np.sqrt(2 * obj_val / mask_count)

            self.history.append((t, gap, obj_val))
            self.step_history.append(gamma)
            self.times.append(time.perf_counter() - t0)
            self.weights_history.append(weights.copy())
            # reduced-memory snapshot
            self.snapshots.append(X.astype(self.snapshot_dtype, copy=False).copy())
            # prune near-zero atoms
            nz = [(a,w) for a,w in zip(atoms,weights) if w>1e-12]
            atoms, weights = map(list, zip(*nz)) if nz else ([], [])

            if gap < self.tol:
                break
            if obj_val < best_obj - self.tol_obj:
                best_obj, no_imp = obj_val, 0
            else:
                no_imp += 1
                if no_imp >= self.patience:
                    break
            # RMSE-based early stop
            if abs(last_rmse - rmse) < 1e-4:
                rmse_no_imp += 1
                if rmse_no_imp >= self.patience:
                    break
            else:
                last_rmse, rmse_no_imp = rmse, 0

        return X
