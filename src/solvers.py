import numpy as np
from scipy.sparse.linalg import svds
from tqdm import trange

class FrankWolfe:
    def __init__(
        self,
        objective,
        lmo_fn,
        tau: float = 1.0,
        max_iter: int = 50,
        tol: float = 1e-4,
        step_method: str = 'vanilla',
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
        raw = -np.sum((X - S) * grad)    # <-- note the leading minus
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

        # 1) Fixed constant gamma
        if self.step_method == 'fixed':
            return float(self.fixed_gamma)

        # 2) Classical diminishing FW
        if self.step_method == 'vanilla':
            return 2.0 / (t + 2)

        # 3) Analytic / Lipschitz-quadratic model step
        if self.step_method == 'analytic':
            num = float(np.sum(grad * d))
            den = float(np.sum((d * self.obj.mask)**2))  # squared masked norm
            if den <= 0:
                # no local curvature → fall back to vanilla
                return 2.0 / (t + 2)
            alpha = -num / den
            return float(np.clip(alpha, 0.0, 1.0))

        # 4) Exact line-search on a fixed grid
        if self.step_method == 'line_search':
            alphas = np.linspace(0.0, 1.0, 20)
            vals   = [self.obj.value(X + a * d) for a in alphas]
            
            return float(alphas[np.argmin(vals)])

        # 5) Armijo / backtracking rule
        if self.step_method == 'armijo':
            # delta in (0,1), beta in (0,1) controls sufficient-decrease
            delta, beta = 0.5, 0.1
            fk = self.obj.value(X)
            gd = float(np.sum(grad * d))
            alpha = 1.0
            # Shrink until we satisfy f(X+alpha d) <= f(X) + beta * alpha * <grad,d>
            while alpha > 1e-8:
                if self.obj.value(X + alpha * d) <= fk + beta * alpha * gd:
                    return alpha
                alpha *= delta
            return 0.0

        # Fallback to vanilla if method name is unrecognized
        return 2.0 / (t + 2)

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


class PairwiseFrankWolfe(FrankWolfe):
    def run(self, X0=None):
        # 0) Initialization
        X = np.zeros_like(self.obj.M_obs) if X0 is None else X0.copy()
        atoms, weights = [], []

        for t in trange(self.max_iter, desc='Pairwise-FW'):
            # 1) Compute gradient and LMO
            grad = self.obj.gradient(X)
            S    = self.lmo(grad, self.tau)

            # 2) Ensure S in active set
            if S not in atoms:
                atoms.append(S);  weights.append(0.0)
            idx_S = atoms.index(S)

            # 3) Away atom selection
            scores   = [np.sum(grad * A) for A in atoms]
            idx_max  = int(np.argmax(scores))
            V        = atoms[idx_max]
            alpha_max= weights[idx_max]

            # 4) Pairwise direction & dual gap
            d   = S - V
            gap = self._dual_gap(X, grad, S)
            self.history.append((t, gap, self.obj.value(X)))
            if gap < self.tol:
                break

            # 5) Choose step‐size (capped by alpha_max)
            gamma = min(self._choose_step(X, grad, S, d, t), alpha_max)

            # 6) In‐place update of X
            X += gamma * d    # <-- just add gamma*(S - V)

            # 7) Update weights
            weights[idx_S]   += gamma
            weights[idx_max] -= gamma

            # 8) Prune zero‐weight atoms
            nz = [(a,w) for (a,w) in zip(atoms, weights) if w > 1e-12]
            if nz:
                atoms, weights = map(list, zip(*nz))
            else:
                atoms, weights = [], []

        return X

