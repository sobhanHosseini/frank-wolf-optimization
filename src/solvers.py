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
    power_method.prev_vec = x.copy()

    left_vec  = y / (norm_y if norm_y > 0 else 1.0)
    right_vec = x.reshape(1, -1)
    sigma     = norm_y
    return left_vec.reshape(-1,1), right_vec, sigma


# ----------------------------------------------------------------
# Atom & LMO via power method
# ----------------------------------------------------------------
class Atom:
    def __init__(self, u: np.ndarray, v: np.ndarray, tau: float):
        self.u   = u
        self.v   = v
        self.tau = tau
    def to_matrix(self):
        return -self.tau * (self.u @ self.v)
    def __eq__(self, other):
        return isinstance(other, Atom) and np.allclose(self.u, other.u) and np.allclose(self.v, other.v)
    def __repr__(self):
        return f"Atom(tau={self.tau}, rank1)"

def nuclear_norm_lmo(gradient: sparse.csr_matrix, tau: float):
    mat_vec = lambda x: gradient @ x
    vec_mat = lambda y: gradient.T @ y
    u, v, _ = power_method(mat_vec, vec_mat, gradient.shape)
    S = -tau * (u @ v)
    if gradient.multiply(S).sum() > 0:
        u = -u
        S = -tau * (u @ v)
    return Atom(u, v, tau)


# ----------------------------------------------------------------
# Objective: sparse gradient
# ----------------------------------------------------------------
class MatrixCompletionObjective:
    def __init__(self, observed_matrix: sparse.csr_matrix, mask: np.ndarray):
        self.observed_matrix = observed_matrix
        self.mask            = mask
        self.rows, self.cols = observed_matrix.nonzero()
        self.data            = observed_matrix.data.copy()

    def value(self, X_obs: np.ndarray) -> float:
        diff = (X_obs - self.observed_matrix.toarray())[self.mask]
        return 0.5 * float(np.dot(diff, diff))

    def gradient(self, X_obs: np.ndarray) -> sparse.csr_matrix:
        residual = X_obs[self.rows, self.cols] - self.data
        return sparse.csr_matrix((residual, (self.rows, self.cols)), shape=X_obs.shape)


# ----------------------------------------------------------------
# Frank-Wolfe (FW) with low-rank factors
# ----------------------------------------------------------------
class FrankWolfe:
    def __init__(self, objective, lmo_function, tau=1.0, max_iter=50,
                 tol=1e-3, abs_tol=1e-6, snapshot_interval=5,
                 step_method='analytic', fixed_step=0.1):
        self.objective        = objective
        self.lmo_function     = lmo_function
        self.tau              = tau
        self.max_iter         = max_iter
        self.tol              = tol
        self.abs_tol          = abs_tol
        self.snapshot_interval = snapshot_interval
        self.step_method      = step_method
        self.fixed_step       = fixed_step
        self.atoms            = []
        self.weights          = []
        self.history          = []
        self.step_history     = []
        self.times            = []
        self.snapshots        = []
        self.snapshot_iters   = []

    def _reconstruct_X_obs(self):
        mask  = self.objective.mask
        X_obs = np.zeros_like(self.objective.observed_matrix.toarray())
        for w, atom in zip(self.weights, self.atoms):
            X_obs[mask] += w * (atom.u @ atom.v.T)[mask]
        return X_obs

    def _dual_gap(self, gradient, S_matrix):
        gap_fw   = -gradient.multiply(S_matrix).sum()
        gap_curr = sum(w * (-atom.to_matrix().multiply(gradient).sum())
                       for w, atom in zip(self.weights, self.atoms))
        return float(max(gap_curr + gap_fw, 0.0))

    def _choose_step(self, gradient, direction, dir_norm_sq, iteration):
        if self.step_method == 'fixed':
            return self.fixed_step
        if self.step_method == 'vanilla':
            return 2.0 / (iteration + 2)
        # analytic
        num = float(np.sum(gradient.multiply(direction)))
        den = float(dir_norm_sq)
        if den > 0:
            return float(np.clip(-num/den, 0.0, 1.0))
        return 2.0 / (iteration + 2)

    def run(self):
        start_time = time.time()
        X_obs       = self._reconstruct_X_obs()
        self.times.append(0.0)
        self.snapshots.append(X_obs.copy())
        self.snapshot_iters.append(0)
        initial_gap = None

        for t in trange(self.max_iter, desc='FW'):
            X_obs    = self._reconstruct_X_obs()
            grad     = self.objective.gradient(X_obs)
            atom     = self.lmo_function(grad, self.tau)
            S_matrix = atom.to_matrix()

            direction   = S_matrix - X_obs
            dir_norm_sq = np.sum((direction * self.objective.mask)**2)
            gap         = self._dual_gap(grad, S_matrix)

            if initial_gap is None:
                initial_gap = gap
            if gap <= self.tol*initial_gap or gap <= self.abs_tol:
                break

            step_size = self._choose_step(grad, direction, dir_norm_sq, t)
            if atom not in self.atoms:
                self.atoms.append(atom)
                self.weights.append(0.0)
            idx = self.atoms.index(atom)
            self.weights[idx] += step_size

            self.history.append((t, gap, self.objective.value(X_obs)))
            self.step_history.append(step_size)
            self.times.append(time.time() - start_time)
            if (t+1)%self.snapshot_interval==0 or t==self.max_iter-1:
                self.snapshots.append(self._reconstruct_X_obs())
                self.snapshot_iters.append(t+1)

        self.snapshots.append(self._reconstruct_X_obs())
        self.snapshot_iters.append(t+1)
        return None


# ----------------------------------------------------------------
# Pairwise FW (PFW)
# ----------------------------------------------------------------
class PairwiseFrankWolfe(FrankWolfe):
    def run(self):
        start_time = time.time()
        X_obs       = self._reconstruct_X_obs()
        self.times          = [0.0]
        self.snapshots      = [X_obs.copy()]
        self.snapshot_iters = [0]
        self.history        = []
        self.step_history   = []
        self.weights_history= []
        initial_gap        = None

        for t in trange(self.max_iter, desc='PFW'):
            X_obs    = self._reconstruct_X_obs()
            grad     = self.objective.gradient(X_obs)
            atom_fw  = self.lmo_function(grad, self.tau)
            if atom_fw not in self.atoms:
                self.atoms.append(atom_fw)
                self.weights.append(0.0)
            idx_fw   = self.atoms.index(atom_fw)

            scores   = [grad.multiply(a.to_matrix()).sum() for a in self.atoms]
            idx_away = int(np.argmax(scores))
            alpha_max= self.weights[idx_away]
            S_fw     = atom_fw.to_matrix()

            if alpha_max <= 0:
                direction   = S_fw - X_obs
                denom       = np.sum((direction*self.objective.mask)**2)
                step0       = self._choose_step(grad, direction, denom, t)
                step_size   = step0
                self.weights[idx_fw] += step_size
            else:
                V_away    = self.atoms[idx_away].to_matrix()
                direction = S_fw - V_away
                denom     = np.sum((direction*self.objective.mask)**2)
                step0     = self._choose_step(grad, direction, denom, t)
                step_size = min(step0, alpha_max)
                self.weights[idx_fw]   += step_size
                self.weights[idx_away] -= step_size

            gap = self._dual_gap(grad, S_fw)
            if initial_gap is None:
                initial_gap = gap
            if gap <= self.tol*initial_gap or gap <= self.abs_tol:
                break

            self.history.append((t, gap, self.objective.value(X_obs)))
            self.step_history.append(step_size)
            self.weights_history.append(self.weights.copy())
            self.times.append(time.time() - start_time)
            if (t+1)%self.snapshot_interval==0 or t==self.max_iter-1:
                self.snapshots.append(self._reconstruct_X_obs())
                self.snapshot_iters.append(t+1)

            prune_thresh = 1e-8
            nz = [(a,w) for a,w in zip(self.atoms,self.weights) if w>prune_thresh]
            if nz:
                self.atoms, self.weights = map(list, zip(*nz))
            else:
                self.atoms, self.weights = [],[]

        self.snapshots.append(self._reconstruct_X_obs())
        self.snapshot_iters.append(t+1)
        return None
