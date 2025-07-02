import numpy as np
from scipy.sparse.linalg import svds
from tqdm import trange

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
        self.mask = mask

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
    Linear Minimization Oracle over the nuclear-norm ball.
    Returns S = +tau * u1 v1^T, where (u1,v1) are top singular vectors of grad.
    """
    try:
        # For f(X)=½‖X–M_obs‖², grad = X–M_obs
        # We want the top singular vectors of +grad.
        u, s, vt = svds(grad, k=1, tol=1e-3, maxiter=200)
        if s[0] < 0:
            raise RuntimeError("Non-positive singular value")
    except Exception:
        # Fall back to full SVD
        U, S, VT = np.linalg.svd(grad, full_matrices=False)
        u, vt = U[:, :1], VT[:1, :]
    # **No leading minus** here**
    return Atom(u, vt, tau)

# ----------------------------------------------------------------
# Rank-1 Atom container
# ----------------------------------------------------------------
class Atom:
    def __init__(self, u: np.ndarray, v: np.ndarray, tau: float):
        self.u = u.reshape(-1,1)
        self.v = v.reshape(1,-1)
        self.tau = tau

    def to_matrix(self) -> np.ndarray:
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
# Frank–Wolfe solver with debug prints
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
        debug: bool = False
    ):
        self.obj = objective
        self.lmo = lmo_fn
        self.tau = tau
        self.max_iter = max_iter
        self.tol = tol
        self.step_method = step_method
        self.fixed_gamma = fixed_gamma
        self.tol_obj = tol_obj
        self.patience = patience
        self.debug = debug
        self.history = []  # (iter, dual_gap, obj_value)

    def _dual_gap(self, X, grad, S):
        # true FW duality gap = ⟨X−S, grad⟩ ≥ 0
        raw = -np.sum((S - X) * grad)       # = np.sum((X-S)*grad)
        return max(raw, 0.0)

    def _choose_step(self, X, grad, S, d, t) -> float:
        if self.step_method == 'fixed':
            return float(self.fixed_gamma)
        if self.step_method == 'vanilla':
            return 2.0 / (t + 2)
        if self.step_method == 'analytic':
            num = float(np.sum(grad * d))
            den = float(np.sum((d * self.obj.mask)**2))
            if den <= 0:
                return 2.0 / (t + 2)
            return float(np.clip(-num / den, 0.0, 1.0))
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

    def run(self, X0=None) -> np.ndarray:
        X = np.zeros_like(self.obj.M_obs) if X0 is None else X0.copy()
        best_obj = float('inf')
        no_improve = 0

        for t in trange(self.max_iter, desc='Frank-Wolfe'):
            grad = self.obj.gradient(X)
            atom = self.lmo(grad, self.tau)
            S = atom.to_matrix()
            d = S - X

            gap = self._dual_gap(X, grad, S)
            obj_val = self.obj.value(X)
            self.history.append((t, gap, obj_val))

            if self.debug and t < 10:
                d_norm = np.linalg.norm(d)
                print(f"[FW] it={t:2d}  gap={gap:.3e}  obj={obj_val:.3e}  ||d||={d_norm:.3e}", end='')

            if gap < self.tol:
                if self.debug and t < 10: print("  stop by gap")
                break
            if obj_val < best_obj - self.tol_obj:
                best_obj = obj_val
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    if self.debug and t < 10: print("  stop by obj")
                    break

            gamma = self._choose_step(X, grad, S, d, t)
            if self.debug and t < 10:
                print(f"  gamma={gamma:.3e}")
            X += gamma * d

        return X

# ----------------------------------------------------------------
# Pairwise Frank–Wolfe solver with debug prints and FW fallback
# ----------------------------------------------------------------
class PairwiseFrankWolfe(FrankWolfe):
    def __init__(self, *args, debug: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.debug = debug

    def run(self, X0=None) -> np.ndarray:
        X = np.zeros_like(self.obj.M_obs) if X0 is None else X0.copy()
        atoms, weights = [], []

        for t in trange(self.max_iter, desc='Pairwise-FW'):
            grad = self.obj.gradient(X)
            atom = self.lmo(grad, self.tau)

            if atom not in atoms:
                atoms.append(atom)
                weights.append(0.0)
            idx_S = atoms.index(atom)

            scores = [np.sum(grad * a.to_matrix()) for a in atoms]
            idx_away = int(np.argmax(scores))
            alpha_max = weights[idx_away]

            if self.debug and t < 10:
                print(f"[PFW] it={t:2d}  atoms={len(atoms)}  idx_S={idx_S}  idx_away={idx_away}  alpha_max={alpha_max:.3e}", end='')

            S = atom.to_matrix()
            if alpha_max <= 0:
                # fallback to full Frank-Wolfe step
                d = S - X
                gamma = self._choose_step(X, grad, S, d, t)
                weights[idx_S] += gamma
            else:
                V = atoms[idx_away].to_matrix()
                d = S - V
                gamma = min(self._choose_step(X, grad, S, d, t), alpha_max)
                weights[idx_S] += gamma
                weights[idx_away] -= gamma

            if self.debug and t < 10:
                print(f"  gamma={gamma:.3e}")

            X += gamma * d

            nz = [(a, w) for a, w in zip(atoms, weights) if w > 1e-12]
            atoms, weights = map(list, zip(*nz)) if nz else ([], [])

            gap = self._dual_gap(X, grad, S)
            obj_val = self.obj.value(X)
            self.history.append((t, gap, obj_val))

            if gap < self.tol:
                break

        return X
