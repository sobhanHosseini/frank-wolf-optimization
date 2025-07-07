import numpy as np
from scipy import sparse
from tqdm import trange
import time

# ----------------------------------------------------------------
# Power method for top singular vector (per Jaggi 2013), with warm‐start
# ----------------------------------------------------------------
def power_method(mat_vec, vec_mat, shape, num_iters=50, tol=1e-6):
    m, n = shape
    # Warm-start: reuse previous right-singular vector if available
    if hasattr(power_method, 'prev_x') and power_method.prev_x.shape == (n,):
        x = power_method.prev_x.copy()
    else:
        x = np.random.randn(n)
    x_norm = np.linalg.norm(x)
    if x_norm > 0:
        x /= x_norm

    for _ in range(num_iters):
        y = mat_vec(x)
        y_norm = np.linalg.norm(y)
        if y_norm == 0:
            break
        x_new = vec_mat(y)
        x_new_norm = np.linalg.norm(x_new)
        if x_new_norm == 0:
            break
        x_new /= x_new_norm
        if np.linalg.norm(x_new - x) < tol:
            x = x_new
            break
        x = x_new
    # Store for next warm-start
    power_method.prev_x = x.copy()

    u = y / (y_norm if y_norm > 0 else 1.0)
    v = x.reshape(1, -1)
    sigma = y_norm
    return u.reshape(-1,1), v, sigma


# ----------------------------------------------------------------
# Rank-1 Atom container & nuclear-norm LMO via power method
# ----------------------------------------------------------------
class Atom:
    def __init__(self, u: np.ndarray, v: np.ndarray, tau: float):
        self.u = u  # (m,1)
        self.v = v  # (1,n)
        self.tau = tau
    def to_matrix(self):
        return -self.tau * (self.u @ self.v)
    def __eq__(self, other):
        return isinstance(other, Atom) and np.allclose(self.u, other.u) and np.allclose(self.v, other.v)
    def __repr__(self):
        return f"Atom(tau={self.tau}, rank1)"

def nuclear_norm_lmo(grad: sparse.csr_matrix, tau: float):
    # Use power method on sparse gradient
    mat_vec = lambda x: grad @ x
    vec_mat = lambda y: grad.T @ y
    u, v, _ = power_method(mat_vec, vec_mat, grad.shape)
    S = -tau * (u @ v)
    # Ensure descent direction
    if np.sum(grad.multiply(S)) > 0:
        u = -u
        S = -tau * (u @ v)
    return Atom(u, v, tau)


# ----------------------------------------------------------------
# Objective: half squared error on observed entries, sparse gradient
# ----------------------------------------------------------------
class MatrixCompletionObjective:
    def __init__(self, M_obs: sparse.csr_matrix, mask: np.ndarray):
        self.M_obs = M_obs            # sparse CSR
        self.mask  = mask            # boolean array
        self.ui, self.ij = M_obs.nonzero()
        self.data = M_obs.data.copy()

    def value(self, X_obs: np.ndarray) -> float:
        diff = (X_obs - self.M_obs.toarray())[self.mask]
        return 0.5 * float(np.dot(diff, diff))

    def gradient(self, X_obs: np.ndarray) -> sparse.csr_matrix:
        # Compute residual only on observed entries
        res = X_obs[self.ui, self.ij] - self.data
        # Build sparse gradient
        return sparse.csr_matrix((res, (self.ui, self.ij)), shape=X_obs.shape)


# ----------------------------------------------------------------
# Frank–Wolfe solver (FW) using low-rank factors
# ----------------------------------------------------------------
class FrankWolfe:
    def __init__(
        self,
        objective,
        lmo_function,
        tau: float = 1.0,
        max_iter: int = 50,
        tol: float = 1e-3,
        step_method: str = 'analytic',
        fixed_step: float = 0.1,
        snapshot_interval: int = 5,
        abs_tol: float = 1e-6
    ):
        # problem data
        self.objective         = objective
        self.lmo_function      = lmo_function
        self.tau               = tau
        self.max_iter          = max_iter
        self.tol               = tol               # relative gap tol
        self.step_method       = step_method       # 'analytic','vanilla','fixed'
        self.fixed_step        = fixed_step        # only if step_method='fixed'
        self.snapshot_interval = snapshot_interval
        self.abs_tol           = abs_tol           # absolute gap tol

        # active‐set representation
        self.atoms   = []  # list of Atom
        self.weights = []  # corresponding weights

        # logging
        self.history        = []  # (iter, gap, obj)
        self.step_history   = []  # chosen γ each step
        self.times          = []  # wall‐clock at each log
        self.snapshots      = []  # X_obs at each snapshot
        self.snapshot_iters = []  # iteration numbers of snapshots

    def _reconstruct_X_obs(self):
        mask = self.objective.mask
        # M_obs is csr‐matrix with same shape
        X_obs = np.zeros_like(self.objective.M_obs.toarray())
        for w, atom in zip(self.weights, self.atoms):
            A = atom.u @ atom.v.T
            X_obs[mask] += w * A[mask]
        return X_obs

    def _dual_gap(self, gradient, S):
        # gap = -⟨S - X, grad⟩
        gap_fw   = -gradient.multiply(S).sum()
        gap_prev = sum(
            w * (-atom.to_matrix().multiply(gradient).sum())
            for w, atom in zip(self.weights, self.atoms)
        )
        return float(max(gap_prev + gap_fw, 0.0))

    def _choose_step(self, gradient, direction, dir_norm_sq, it):
        if self.step_method == 'fixed':
            return self.fixed_step
        if self.step_method == 'vanilla':
            return 2.0 / (it + 2)
        # analytic line‐search for quadratic loss on Ω
        num = float(np.sum(gradient.multiply(direction)))
        den = float(dir_norm_sq)
        if den > 0:
            return float(np.clip(-num / den, 0.0, 1.0))
        return 2.0 / (it + 2)

    def run(self):
        start = time.time()

        # initial X_obs = 0
        X_obs = self._reconstruct_X_obs()
        self.times.append(0.0)
        self.snapshots.append(X_obs.copy())
        self.snapshot_iters.append(0)

        initial_gap = None

        for t in trange(self.max_iter, desc='FW'):
            X_obs      = self._reconstruct_X_obs()
            grad       = self.objective.gradient(X_obs)
            atom       = self.lmo_function(grad, self.tau)
            S          = atom.to_matrix()

            direction   = S - X_obs
            dir_norm_sq = np.sum((direction * self.objective.mask)**2)
            gap         = self._dual_gap(grad, S)

            if initial_gap is None:
                initial_gap = gap
            if gap <= self.tol * initial_gap or gap <= self.abs_tol:
                break

            step = self._choose_step(grad, direction, dir_norm_sq, t)

            # add atom if new
            if atom not in self.atoms:
                self.atoms.append(atom)
                self.weights.append(0.0)
            idx = self.atoms.index(atom)
            self.weights[idx] += step

            obj_val = self.objective.value(X_obs)
            self.history.append((t, gap, obj_val))
            self.step_history.append(step)
            self.times.append(time.time() - start)

            if (t + 1) % self.snapshot_interval == 0 or t == self.max_iter - 1:
                self.snapshots.append(self._reconstruct_X_obs())
                self.snapshot_iters.append(t + 1)

        # final snapshot at termination
        self.snapshots.append(self._reconstruct_X_obs())
        self.snapshot_iters.append(min(t + 1, self.max_iter))
        return None

# ----------------------------------------------------------------
# Pairwise Frank-Wolfe solver (PFW) with factor-based X
# ----------------------------------------------------------------
class PairwiseFrankWolfe(FrankWolfe):
    def run(self):
        start_time  = time.time()
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
            gradient = self.objective.gradient(X_obs)
            atom_fw  = self.lmo_function(gradient, self.tau)
            if atom_fw not in self.atoms:
                self.atoms.append(atom_fw)
                self.weights.append(0.0)
            idx_fw    = self.atoms.index(atom_fw)

            # select away atom by max inner product
            scores  = [gradient.multiply(atom.to_matrix()).sum() for atom in self.atoms]
            idx_away = int(np.argmax(scores))
            alpha_max= self.weights[idx_away]
            S_fw     = atom_fw.to_matrix()

            if alpha_max <= 0:
                # Frank-Wolfe move
                direction   = S_fw - X_obs
                norm_sq     = np.sum((direction * self.objective.mask)**2)
                step_size0  = self._choose_step(gradient, direction, norm_sq, t)
                step_size   = step_size0
                self.weights[idx_fw] += step_size
            else:
                # Away move
                V_away      = self.atoms[idx_away].to_matrix()
                direction   = S_fw - V_away
                norm_sq     = np.sum((direction * self.objective.mask)**2)
                step_size0  = self._choose_step(gradient, direction, norm_sq, t)
                step_size   = min(step_size0, alpha_max)
                self.weights[idx_fw]   += step_size
                self.weights[idx_away] -= step_size

            gap = self._dual_gap(gradient, S_fw)
            if initial_gap is None:
                initial_gap = gap
            if gap <= self.tol * initial_gap or gap <= self.abs_tol:
                break

            self.history.append((t, gap, self.objective.value(X_obs)))
            self.step_history.append(step_size)
            self.weights_history.append(self.weights.copy())
            self.times.append(time.time() - start_time)

            if (t + 1) % self.snapshot_interval == 0 or t == self.max_iter - 1:
                self.snapshots.append(self._reconstruct_X_obs())
                self.snapshot_iters.append(t + 1)

            # prune small weights
            prune_threshold = 1e-8
            non_zero = [(a,w) for a,w in zip(self.atoms, self.weights) if w > prune_threshold]
            if non_zero:
                self.atoms, self.weights = map(list, zip(*non_zero))
            else:
                self.atoms, self.weights = [], []

        # final snapshot
        self.snapshots.append(self._reconstruct_X_obs())
        self.snapshot_iters.append(t + 1)
        return None