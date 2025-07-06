import os
import time
import numpy as np
import matplotlib.pyplot as plt

from utils import load_dataset, evaluate, approximate_nuclear_norm
from solvers import (
    MatrixCompletionObjective,
    nuclear_norm_lmo,
    FrankWolfe,
    PairwiseFrankWolfe,
)

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
config = {
    'datasets':           ['ml-100k', 'jester2'],
    'test_fraction':      0.2,
    'seed':               42,
    'tau_scale':          1.0,
    'tau_approx_k':       10,
    'max_iter':           200,
    'tol':                1e-2,   # relative‐gap tolerance
    'snapshot_interval':  10,
    'save_plots':         False,
    'plot_dir':           'plots',
}

if config['save_plots']:
    os.makedirs(config['plot_dir'], exist_ok=True)

# ----------------------------------------------------------------------------
# Utility
# ----------------------------------------------------------------------------
def subsample_indices(n, num=20):
    if n <= num:
        return np.arange(n)
    return np.unique(np.linspace(0, n - 1, num, dtype=int))

# ----------------------------------------------------------------------------
# Experiment runner
# ----------------------------------------------------------------------------
class ExperimentRunner:
    def __init__(self, cfg):
        self.cfg          = cfg
        self.results      = {}  # results[dataset][solver] → dict of curves
        self.summary_rows = []  # (dataset, solver, rmse_tr, rmse_te, nrm_tr, nrm_te, iters)

    def run(self):
        for ds in self.cfg['datasets']:
            print(f"\n=== Dataset: {ds} ===")
            # load train/test split
            M_obs, mask_tr, mask_te, M_true = load_dataset(
                name=ds,
                test_fraction=self.cfg['test_fraction'],
                seed=self.cfg['seed']
            )

            # center by train‐mean
            mu_tr = M_true[mask_tr].mean()
            M_obs_c = M_obs.copy()
            M_obs_c[mask_tr] -= mu_tr

            # compute rating range for normalization
            vals = M_true[mask_tr]
            rmin, rmax = float(vals.min()), float(vals.max())
            rating_range = rmax - rmin

            # setup objective and τ
            obj     = MatrixCompletionObjective(M_obs_c, mask_tr)
            def_tau = approximate_nuclear_norm(M_obs_c, k=self.cfg['tau_approx_k'])
            tau     = self.cfg['tau_scale'] * def_tau
            print(f"μ_train={mu_tr:.3f}, τ≈{def_tau:.3f} → using τ={tau:.3f}")

            ds_res = {}
            for solver_name, Solver in [('FW', FrankWolfe), ('PFW', PairwiseFrankWolfe)]:
                print(f"--- {solver_name} ---")
                solver = Solver(
                    objective=obj,
                    lmo_fn=nuclear_norm_lmo,
                    tau=tau,
                    max_iter=self.cfg['max_iter'],
                    tol=self.cfg['tol'],
                    snapshot_interval=self.cfg['snapshot_interval']
                )
                t0 = time.time()
                solver.run()
                dur = time.time() - t0
                n_iters = len(solver.history)
                print(f"{solver_name} done in {dur:.1f}s, iterations = {n_iters}")

                # align snapshots (drop the t=0 snapshot)
                snaps     = solver.snapshots[1:n_iters+1]
                snap_iters= np.array(solver.snapshot_iters[1:n_iters+1])

                # collect curves
                gaps    = np.array(solver.gap_history[1:n_iters+1])
                objs    = np.array([h[2] for h in solver.history])
                times   = np.array(solver.times[1:n_iters+1])
                steps   = np.array(solver.step_history[:n_iters])

                rmse_tr = np.array([
                    evaluate(M_true, Xk + mu_tr, mask_tr)
                    for Xk in snaps
                ])
                rmse_te = np.array([
                    evaluate(M_true, Xk + mu_tr, mask_te)
                    for Xk in snaps
                ])

                ds_res[solver_name] = {
                    'snap_iters':   snap_iters,
                    'gap':          gaps,
                    'obj_vals':     objs,
                    'times':        times,
                    'step_history': steps,
                    'rmse_train':   rmse_tr,
                    'rmse_test':    rmse_te,
                }

                # summary
                final_tr, final_te = rmse_tr[-1], rmse_te[-1]
                nrm_tr = final_tr / rating_range
                nrm_te = final_te / rating_range
                self.summary_rows.append(
                    (ds, solver_name,
                     float(final_tr), float(final_te),
                     float(nrm_tr), float(nrm_te),
                     n_iters)
                )

            self.results[ds] = ds_res

        return self.results, self.summary_rows

# ----------------------------------------------------------------------------
# Simple summary print
# ----------------------------------------------------------------------------
if __name__ == '__main__':
    runner       = ExperimentRunner(config)
    results, summary_rows = runner.run()

    # print combined summary
    header = f"{'Dataset':10s}{'Solver':10s}{'RMSE_tr':>8s}{'RMSE_te':>8s}" \
             f"{'NRMSE_tr':>8s}{'NRMSE_te':>8s}{'Iters':>6s}"
    print("\n=== Combined Summary ===")
    print(header)
    print('-'*len(header))
    for ds, solver, tr, te, ntr, nte, it in summary_rows:
        print(f"{ds:10s}{solver:10s}{tr:8.4f}{te:8.4f}"
              f"{ntr:8.4f}{nte:8.4f}{it:6d}")
