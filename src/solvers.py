# solvers.py

import numpy as np
import time
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

# ----------------------------------------------------------------
# Problem definition: sparse‐aware squared‐error objective
# ----------------------------------------------------------------
class MatrixCompletionObjective:
    def __init__(self, M_obs, mask):
        """
        M_obs : dense ndarray or scipy.sparse.csr_matrix of observed entries.
        mask  : boolean ndarray of same shape, True on training entries.
        """
        if not isinstance(M_obs, csr_matrix):
            M_obs = csr_matrix(M_obs)
        self.M_obs = M_obs.tocsr()
        self.mask  = mask
        coo      = self.M_obs.tocoo()
        self.rows = coo.row
        self.cols = coo.col
        self.data = coo.data

    def value(self, X):
        pred = X[self.rows, self.cols]
        diff = pred - self.data
        return 0.5 * float(np.dot(diff, diff))

    def gradient(self, X):
        pred = X[self.rows, self.cols]
        diff = pred - self.data
        return csr_matrix((diff, (self.rows, self.cols)), shape=X.shape)


# ----------------------------------------------------------------
# Nuclear‐norm‐ball LMO (handles sparse grad)
# ----------------------------------------------------------------
def nuclear_norm_lmo(grad, tau):
    """
    grad: can be dense or scipy.sparse matrix.
    Returns best rank-1 atom S = -tau * u v^T.
    """
    try:
        u, s, vt = svds(grad, k=1, tol=1e-3, maxiter=200)
        if s[0] < 0:
            raise RuntimeError
        u  = u[:, :1]
        vt = vt[:1, :]
    except Exception:
        G = grad.toarray() if hasattr(grad, "toarray") else np.array(grad)
        U, S, VT = np.linalg.svd(G, full_matrices=False)
        u, vt    = U[:, :1], VT[:1, :]
    return Atom(u, vt, tau)


class Atom:
    def __init__(self, u, v, tau):
        self.u   = u.reshape(-1, 1)
        self.v   = v.reshape(1, -1)
        self.tau = tau

    def to_matrix(self):
        return -self.tau * (self.u @ self.v)

    def __repr__(self):
        return f"Atom(tau={self.tau:.3e}, u={self.u.shape}, v={self.v.shape})"


# ----------------------------------------------------------------
# Frank–Wolfe with multiple step‐size rules, curvature & relative‐gap stopping
# ----------------------------------------------------------------
class FrankWolfe:
    def __init__(
        self,
        objective,
        lmo_fn,
        tau,
        max_iter=200,
        tol=1e-2,
        step_method='analytic',
        fixed_gamma=0.1,
        snapshot_interval=10
    ):
        print('test....')
        self.obj        = objective
        self.lmo        = lmo_fn
        self.tau        = tau
        self.max_iter   = max_iter
        self.tol        = tol               # relative gap tolerance
        self.step_method = step_method      # 'fixed','vanilla','analytic'
        self.fixed_gamma = fixed_gamma
        self.snap_int   = snapshot_interval

        # histories
        self.history         = []  # (iter, gap, obj)
        self.gap_history     = []
        self.rel_gap_history = []
        self.curvature_hist  = []
        self.step_history    = []
        self.snapshots       = []
        self.snapshot_iters  = []
        self.times           = []

    def _dual_gap(self, X, grad, S):
        raw = -np.sum((S - X) * grad.toarray())
        return float(max(raw, 0.0))

    def _choose_step(self, X, grad, S, d, t):
        if self.step_method == 'fixed':
            return float(self.fixed_gamma)
        if self.step_method == 'vanilla':
            return 2.0 / (t + 2)
        # analytic exact line-search for quadratic loss
        if self.step_method == 'analytic':
            num = float(np.sum(grad.toarray() * d))
            den = float(np.sum((d[self.obj.rows, self.obj.cols])**2))
            if den <= 0:
                return 2.0 / (t + 2)
            return float(np.clip(-num / den, 0.0, 1.0))
        # fallback to vanilla
        return 2.0 / (t + 2)

    def run(self, X0=None):
        X = np.zeros_like(self.obj.M_obs.toarray()) if X0 is None else X0.copy()
        t0 = time.perf_counter()

        # initial gap
        grad  = self.obj.gradient(X)
        atom0 = self.lmo(grad, self.tau)
        gap0  = self._dual_gap(X, grad, atom0.to_matrix())

        self.gap_history.append(gap0)
        self.rel_gap_history.append(1.0)
        self.snapshots.append(X.copy())
        self.snapshot_iters.append(0)
        self.times.append(0.0)

        for t in range(1, self.max_iter + 1):
            grad  = self.obj.gradient(X)
            atom  = self.lmo(grad, self.tau)
            S     = atom.to_matrix()
            d     = S - X

            gap   = self._dual_gap(X, grad, S)
            objv  = self.obj.value(X)
            self.history.append((t, gap, objv))
            self.gap_history.append(gap)
            self.rel_gap_history.append(gap / gap0)

            if gap <= self.tol * gap0:
                break

            gamma = self._choose_step(X, grad, S, d, t)
            self.step_history.append(gamma)

            # curvature
            f0  = objv
            f1  = self.obj.value(X + gamma * d)
            num = f1 - f0 - gamma * np.sum(grad.toarray() * d)
            curv= 2 * num / (gamma * gamma + 1e-16)
            self.curvature_hist.append(curv)

            X += gamma * d
            self.times.append(time.perf_counter() - t0)

            if t % self.snap_int == 0 or t == self.max_iter:
                self.snapshots.append(X.copy())
                self.snapshot_iters.append(t)

        return X


# ----------------------------------------------------------------
# Pairwise Frank–Wolfe
# ----------------------------------------------------------------
class PairwiseFrankWolfe(FrankWolfe):
    def run(self, X0=None):
        X = np.zeros_like(self.obj.M_obs.toarray()) if X0 is None else X0.copy()
        t0 = time.perf_counter()

        self.atoms, self.weights = [], []
        self.history         = []
        self.gap_history     = []
        self.rel_gap_history = []
        self.curvature_hist  = []
        self.step_history    = []
        self.snapshots       = [X.copy()]
        self.snapshot_iters  = [0]
        self.times           = [0.0]

        # initial gap
        grad  = self.obj.gradient(X)
        atom0 = self.lmo(grad, self.tau)
        gap0  = self._dual_gap(X, grad, atom0.to_matrix())
        self.gap_history.append(gap0)
        self.rel_gap_history.append(1.0)

        for t in range(1, self.max_iter + 1):
            grad = self.obj.gradient(X)
            atom = self.lmo(grad, self.tau)
            if atom not in self.atoms:
                self.atoms.append(atom); self.weights.append(0.0)
            idx_S = self.atoms.index(atom)

            # away direction
            scores   = [np.sum(grad.toarray() * a.to_matrix()) for a in self.atoms]
            idx_away = int(np.argmax(scores))
            alpha_max= self.weights[idx_away]

            if alpha_max <= 0:
                d = atom.to_matrix() - X
            else:
                V = self.atoms[idx_away].to_matrix()
                d = atom.to_matrix() - V

            gap   = self._dual_gap(X, grad, atom.to_matrix())
            objv  = self.obj.value(X)
            self.history.append((t, gap, objv))
            self.gap_history.append(gap)
            self.rel_gap_history.append(gap / gap0)

            if gap <= self.tol * gap0:
                break

            gamma = self._choose_step(X, grad, atom.to_matrix(), d, t)
            if alpha_max > 0:
                gamma = min(gamma, alpha_max)
            self.step_history.append(gamma)

            # curvature
            f0  = objv
            f1  = self.obj.value(X + gamma * d)
            num = f1 - f0 - gamma * np.sum(grad.toarray() * d)
            curv= 2 * num / (gamma * gamma + 1e-16)
            self.curvature_hist.append(curv)

            if alpha_max > 0:
                self.weights[idx_away] -= gamma
            self.weights[idx_S] += gamma
            X += gamma * d
            self.times.append(time.perf_counter() - t0)

            # prune atoms
            nz = [(a,w) for a,w in zip(self.atoms, self.weights) if w > 1e-12]
            if nz:
                self.atoms, self.weights = zip(*nz)
            else:
                self.atoms, self.weights = [], []

            if t % self.snap_int == 0 or t == self.max_iter:
                self.snapshots.append(X.copy())
                self.snapshot_iters.append(t)

        return X
