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
    'steps':              ['analytic'],        # step‐size rules to try
    'test_fraction':      0.2,
    'seed':               42,
    'tau_scale':          1.0,
    'tau_approx_k':       10,
    'max_iter':           200,
    'tol':                1e-2,               # relative‐gap tolerance
    'fixed_gamma':        0.1,                # only used if step='fixed'
    'snapshot_interval':  5,
    'save_plots':         False,
    'plot_dir':           'plots',
}

if config['save_plots']:
    os.makedirs(config['plot_dir'], exist_ok=True)


def subsample_indices(n, num=20):
    if n <= num:
        return np.arange(n)
    return np.unique(np.linspace(0, n - 1, num, dtype=int))


class ExperimentRunner:
    def __init__(self, cfg):
        self.cfg          = cfg
        self.results      = {}
        self.summary_rows = []

    def run(self):
        for dataset in self.cfg['datasets']:
            print(f"\n=== Dataset: {dataset} ===")
            # load data
            M_obs, mask_tr, mask_te, M_true = load_dataset(
                name=dataset,
                test_fraction=self.cfg['test_fraction'],
                seed=self.cfg['seed']
            )

            # center by training mean
            mu_tr = M_true[mask_tr].mean()
            M_obs_c = M_obs.copy()
            M_obs_c[mask_tr] -= mu_tr

            # nuclear‐norm parameter τ
            def_tau = approximate_nuclear_norm(M_obs_c, k=self.cfg['tau_approx_k'])
            tau     = self.cfg['tau_scale'] * def_tau
            print(f"μ_train={mu_tr:.3f}, τ≈{def_tau:.3f} → using τ={tau:.3f}")

            # rating range for normalization
            vals         = M_true[mask_tr]
            rating_range = float(vals.max() - vals.min())

            # objective
            obj = MatrixCompletionObjective(M_obs_c, mask_tr)

            ds_res = {}
            for solver_name, Solver in [('FW', FrankWolfe), ('PFW', PairwiseFrankWolfe)]:
                solver_map = {}
                for step in self.cfg['steps']:
                    print(f"--- {solver_name}-{step} ---")
                    # instantiate with positional LMO
                    solver = Solver(
                        obj,                            # objective
                        nuclear_norm_lmo,               # LMO function
                        tau,                            # nuclear‐norm bound
                        self.cfg['max_iter'],           # max iterations
                        self.cfg['tol'],                # relative‐gap tolerance
                        step,                           # step_method: 'analytic','vanilla','fixed'
                        self.cfg['fixed_gamma'],        # only for 'fixed'
                        self.cfg['snapshot_interval']   # snapshot every so many iters
                    )
                    start = time.time()
                    solver.run()
                    duration = time.time() - start
                    n_iters  = len(solver.history)
                    print(f"Finished {solver_name}-{step} in {duration:.1f}s, iters={n_iters}")

                    # align snapshots (drop t=0)
                    snaps      = solver.snapshots[1 : n_iters+1]
                    snap_iters = solver.snapshot_iters[1 : n_iters+1]

                    # compute RMSE on train/test at each snapshot
                    rmse_tr = np.array([
                        evaluate(M_true, Xk + mu_tr, mask_tr)
                        for Xk in snaps
                    ])
                    rmse_te = np.array([
                        evaluate(M_true, Xk + mu_tr, mask_te)
                        for Xk in snaps
                    ])

                    solver_map[step] = {
                        'snap_iters': snap_iters,
                        'rmse_train': rmse_tr,
                        'rmse_test':  rmse_te,
                        'gap':        np.array(solver.gap_history[1 : n_iters+1]),
                        'obj_vals':   np.array([h[2] for h in solver.history]),
                        'times':      np.array(solver.times[1 : n_iters+1]),
                        'step_history': np.array(solver.step_history[:n_iters]),
                    }

                    # summary
                    final_tr, final_te = rmse_tr[-1], rmse_te[-1]
                    ntr = final_tr / rating_range
                    nte = final_te / rating_range
                    self.summary_rows.append(
                        (dataset, f"{solver_name}-{step}",
                         final_tr, final_te, ntr, nte, n_iters)
                    )

                ds_res[solver_name] = solver_map
            self.results[dataset] = ds_res

        return self.results, self.summary_rows


class ExperimentPlotter:
    def __init__(self, results, summary_rows, cfg):
        self.results      = results
        self.summary_rows = summary_rows
        self.cfg          = cfg

    def plot(self):
        for dataset, data in self.results.items():
            print(f"\n=== Results for {dataset} ===")
            # Simple summary print
            rows = [r for r in self.summary_rows if r[0]==dataset]
            header = f"{'Solver-Step':20s}{'RMSE_tr':>8s}{'RMSE_te':>8s}{'NRMSE_tr':>8s}{'NRMSE_te':>8s}{'Iters':>6s}"
            print(header)
            print('-'*len(header))
            for _, solstep, tr, te, ntr, nte, it in rows:
                print(f"{solstep:20s}{tr:8.4f}{te:8.4f}{ntr:8.4f}{nte:8.4f}{it:6d}")

            # (You can re-add plotting code here if you like, using
            #  data[solver]['snap_iters'], data[solver]['gap'], etc.)

        # Combined summary
        if len(self.cfg['datasets'])>1:
            print("\n=== Combined Summary ===")
            header = f"{'Dataset':10s}{'Solver':10s}{'RMSE_tr':>8s}{'RMSE_te':>8s}{'NRMSE_tr':>8s}{'NRMSE_te':>8s}{'Iters':>6s}"
            print(header)
            print('-'*len(header))
            for ds, sol, tr, te, ntr, nte, it in self.summary_rows:
                print(f"{ds:10s}{sol:10s}{tr:8.4f}{te:8.4f}{ntr:8.4f}{nte:8.4f}{it:6d}")


if __name__ == '__main__':
    runner  = ExperimentRunner(config)
    results, summary_rows = runner.run()
    plotter = ExperimentPlotter(results, summary_rows, config)
    plotter.plot()
