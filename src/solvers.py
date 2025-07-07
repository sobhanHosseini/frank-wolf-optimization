# solvers.py

import numpy as np
from scipy import sparse
from tqdm import trange
import time

# ----------------------------------------------------------------
# Power method for top singular vector (Jaggi 2013), with warm-start
# ----------------------------------------------------------------
def power_method(mat_vec, vec_mat, shape, num_iters=50, tol=1e-6):
    m, n = shape
    # reuse previous vector if available
    if hasattr(power_method, 'prev_vec') and power_method.prev_vec.shape == (n,):
        x = power_method.prev_vec.copy()
    else:
        x = np.random.randn(n)
    norm_x = np.linalg.norm(x)
    if norm_x > 0:
        x /= norm_x

    for _ in range(num_iters):
        y = mat_vec(x)
        norm_y = np.linalg.norm(y)
        if norm_y == 0:
            break
        x_new = vec_mat(y)
        norm_x_new = np.linalg.norm(x_new)
        if norm_x_new == 0:
            break
        x_new /= norm_x_new
        if np.linalg.norm(x_new - x) < tol:
            x = x_new
            break
        x = x_new

    # store for next warm-start
    power_method.prev_vec = x.copy()

    u = y / (norm_y if norm_y > 0 else 1.0)
    v = x.reshape(1, -1)
    return u.reshape(-1,1), v, norm_y


# ----------------------------------------------------------------
# Rank-1 Atom container & nuclear-norm LMO via power method
# ----------------------------------------------------------------
class Atom:
    def __init__(self, u: np.ndarray, v: np.ndarray, tau: float):
        self.u   = u  # shape (m,1)
        self.v   = v  # shape (1,n)
        self.tau = tau

    def to_matrix(self):
        return -self.tau * (self.u @ self.v)

    def __eq__(self, other):
        return isinstance(other, Atom) and np.allclose(self.u, other.u) and np.allclose(self.v, other.v)

    def __repr__(self):
        return f"Atom(tau={self.tau}, rank1)"

def nuclear_norm_lmo(gradient: sparse.csr_matrix, tau: float):
    """
    Returns the rank-1 atom S = -τ u v^T solving
    min ⟨gradient, Z⟩ s.t. ||Z||_* ≤ τ,
    via the power method for the top singular vector.
    """
    mat_vec = lambda x: gradient @ x
    vec_mat = lambda y: gradient.T @ y
    u, v, _ = power_method(mat_vec, vec_mat, gradient.shape)
    atom = Atom(u, v, tau)

    # ensure descent direction: ⟨gradient, S⟩ <= 0
    rows, cols = gradient.nonzero()
    u_val = u[rows,0]
    v_val = v[0,cols]
    S_data = -tau * (u_val * v_val)
    if np.dot(S_data, gradient.data) > 0:
        atom.u = -atom.u

    return atom


# ----------------------------------------------------------------
# MatrixCompletionObjective: holds sparse M_obs and mask
# ----------------------------------------------------------------
class MatrixCompletionObjective:
    def __init__(self, observed_matrix: sparse.csr_matrix, mask: np.ndarray):
        self.observed_matrix = observed_matrix  # sparse CSR
        self.mask            = mask            # boolean array
        self.rows, self.cols = observed_matrix.nonzero()
        self.data            = observed_matrix.data.copy()
        self.shape           = observed_matrix.shape


# ----------------------------------------------------------------
# Frank–Wolfe (FW) with incremental observed‐entry updates
# ----------------------------------------------------------------
class FrankWolfe:
    def __init__(self, objective, lmo_function, tau=1.0, max_iter=50,
                 tol=1e-3, abs_tol=1e-6, snapshot_interval=5,
                 step_method='analytic', fixed_step=0.1):
        print("test...")
        self.objective        = objective
        self.lmo_function     = lmo_function
        self.tau              = tau
        self.max_iter         = max_iter
        self.tol              = tol
        self.abs_tol          = abs_tol
        self.snapshot_interval = snapshot_interval
        self.step_method      = step_method   # 'analytic', 'vanilla', or 'fixed'
        self.fixed_step       = fixed_step

        # Precompute observed-entry indices & storage
        self.rows, self.cols = objective.rows, objective.cols
        self.data            = objective.data.copy()     # true M_obs data
        self.obs_data        = np.zeros_like(self.data)  # current X on mask

        # Histories for diagnostics
        self.history        = []   # (iter, gap, obj_val)
        self.step_history   = []
        self.times          = []
        self.snapshots      = []
        self.snapshot_iters = []

    def _dual_gap(self, d_data, res):
        # gap = -⟨d, ∇f⟩ where ∇f on mask = residual = obs_data - data
        return float(max(-np.dot(d_data, res), 0.0))

    def _choose_step(self, res, d_data, iteration):
        if self.step_method == 'fixed':
            return self.fixed_step
        if self.step_method == 'vanilla':
            return 2.0 / (iteration + 2)
        # analytic for quadratic loss
        num = float(np.dot(res, d_data))
        den = float(np.dot(d_data, d_data))
        if den > 0:
            return float(np.clip(-num/den, 0.0, 1.0))
        return 2.0 / (iteration + 2)

    def run(self):
        m, n = self.objective.shape
        start = time.perf_counter()

        # initial snapshot (X=0)
        X2d = np.zeros((m, n))
        self.times.append(0.0)
        self.snapshots.append(X2d.copy())
        self.snapshot_iters.append(0)
        initial_gap = None

        for t in trange(self.max_iter, desc='FW'):
            # residual and sparse gradient from obs_data
            res = self.obs_data - self.data
            grad = sparse.csr_matrix((res, (self.rows, self.cols)), shape=(m, n))

            # LMO atom and compute S_data on mask
            atom = self.lmo_function(grad, self.tau)
            u_val = atom.u[self.rows,0]
            v_val = atom.v[0,self.cols]
            S_data = -self.tau * (u_val * v_val)

            # direction and dual gap
            d_data = S_data - self.obs_data
            gap    = self._dual_gap(d_data, res)
            if initial_gap is None:
                initial_gap = gap
            if gap <= self.tol * initial_gap or gap <= self.abs_tol:
                break

            # step-size and update
            step_size = self._choose_step(res, d_data, t)
            self.obs_data += step_size * d_data

            # record objective value: ½||res||²
            obj_val = 0.5 * np.dot(self.obs_data - self.data, self.obs_data - self.data)
            self.history.append((t, gap, obj_val))
            self.step_history.append(step_size)
            self.times.append(time.perf_counter() - start)

            # snapshot
            if (t+1) % self.snapshot_interval == 0 or t == self.max_iter-1:
                X2d = np.zeros((m, n))
                X2d[self.rows, self.cols] = self.obs_data
                self.snapshots.append(X2d.copy())
                self.snapshot_iters.append(t+1)

        # final snapshot
        X2d = np.zeros((m, n))
        X2d[self.rows, self.cols] = self.obs_data
        self.snapshots.append(X2d.copy())
        self.snapshot_iters.append(t+1)

        return None


# ----------------------------------------------------------------
# Pairwise Frank–Wolfe (PFW) with incremental updates
# ----------------------------------------------------------------
class PairwiseFrankWolfe(FrankWolfe):
    def run(self):
        m, n = self.objective.shape
        start = time.perf_counter()

        # initial state
        X2d = np.zeros((m, n))
        self.times          = [0.0]
        self.snapshots      = [X2d.copy()]
        self.snapshot_iters = [0]
        self.history        = []
        self.step_history   = []
        self.weights_history= []
        initial_gap        = None

        # ensure atoms/weights lists exist
        if not hasattr(self, 'atoms'):
            self.atoms = []
            self.weights = []

        for t in trange(self.max_iter, desc='PFW'):
            res  = self.obs_data - self.data
            grad = sparse.csr_matrix((res, (self.rows, self.cols)), shape=(m, n))

            # Frank-Wolfe atom
            atom_fw = self.lmo_function(grad, self.tau)
            if atom_fw not in self.atoms:
                self.atoms.append(atom_fw)
                self.weights.append(0.0)
            idx_fw = self.atoms.index(atom_fw)

            # compute away atom
            scores = []
            for atom in self.atoms:
                u_val = atom.u[self.rows,0]
                v_val = atom.v[0,self.cols]
                A_data = -atom.tau * (u_val * v_val)
                scores.append(np.dot(res, A_data))
            idx_away  = int(np.argmax(scores))
            alpha_max = self.weights[idx_away]

            # compute S_data for FW atom
            u_val_fw = atom_fw.u[self.rows,0]
            v_val_fw = atom_fw.v[0,self.cols]
            S_data   = -self.tau * (u_val_fw * v_val_fw)

            # determine direction and step-size
            if alpha_max <= 0:
                # pure FW move
                d_data = S_data - self.obs_data
                step0  = self._choose_step(res, d_data, t)
                step_size = step0
                self.weights[idx_fw] += step_size
            else:
                # away move
                atom_away = self.atoms[idx_away]
                u_val_away = atom_away.u[self.rows,0]
                v_val_away = atom_away.v[0,self.cols]
                V_data     = -self.tau * (u_val_away * v_val_away)
                d_data     = S_data - V_data
                step0      = self._choose_step(res, d_data, t)
                step_size  = min(step0, alpha_max)
                self.weights[idx_fw]   += step_size
                self.weights[idx_away] -= step_size

            # update X on mask
            self.obs_data += step_size * d_data

            # dual gap and stopping
            gap = self._dual_gap(d_data, res)
            if initial_gap is None:
                initial_gap = gap
            if gap <= self.tol * initial_gap or gap <= self.abs_tol:
                break

            # record
            obj_val = 0.5 * np.dot(self.obs_data - self.data, self.obs_data - self.data)
            self.history.append((t, gap, obj_val))
            self.step_history.append(step_size)
            self.weights_history.append(self.weights.copy())
            self.times.append(time.perf_counter() - start)

            # snapshot
            if (t+1) % self.snapshot_interval == 0 or t == self.max_iter-1:
                X2d = np.zeros((m, n))
                X2d[self.rows, self.cols] = self.obs_data
                self.snapshots.append(X2d.copy())
                self.snapshot_iters.append(t+1)

            # prune atoms with near-zero weight
            prune = [(a,w) for a,w in zip(self.atoms, self.weights) if w > 1e-8]
            if prune:
                self.atoms, self.weights = map(list, zip(*prune))
            else:
                self.atoms, self.weights = [], []

        # final snapshot
        X2d = np.zeros((m, n))
        X2d[self.rows, self.cols] = self.obs_data
        self.snapshots.append(X2d.copy())
        self.snapshot_iters.append(t+1)

        return None
