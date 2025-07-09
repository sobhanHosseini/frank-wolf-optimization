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
    'datasets':           ['ml-100k'],
    'steps':              ['analytic', 'vanilla', 'fixed'],
    'test_fraction':      0.2,
    'seed':               42,
    'tau_scale':          1.0,
    'max_iter':           200,
    'tol':                1e-2,
    'snapshot_interval':  5,
    'save_plots':         False,
    'plot_dir':           'plots',
    'tau_approx_k':       10,
    'fixed_step':         0.1,
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
            M_obs, mask_train, mask_test, M_true = load_dataset(
                name=dataset,
                test_fraction=self.cfg['test_fraction'],
                seed=self.cfg['seed']
            )

            mu_tr = M_true[mask_train].mean()
            M_obs_centered = M_obs.copy()
            M_obs_centered.data -= mu_tr

            vals = M_true[mask_train]
            rating_range = float(vals.max() - vals.min())
            y_tr   = M_true[mask_train] - mu_tr
            y_te   = M_true[mask_test]  - mu_tr
            SST_tr = np.sum(y_tr**2)
            SST_te = np.sum(y_te**2)

            obj     = MatrixCompletionObjective(M_obs_centered, mask_train)
            def_tau = approximate_nuclear_norm(
                M_obs_centered, k=self.cfg['tau_approx_k']
            )
            tau     = self.cfg['tau_scale'] * def_tau
            print(f"μ_train={mu_tr:.3f}, τ≈{def_tau:.3f} → using τ={tau:.3f}")

            ds_results = {}

            # *** FIXED: use solver_name as key, not dataset ***
            for solver_name, Solver in [('FW', FrankWolfe), ('PFW', PairwiseFrankWolfe)]:
                solver_map = {}
                for step in self.cfg['steps']:
                    print(f"--- {solver_name}-{step} ---")
                    solver = Solver(
                        objective=obj,
                        lmo_function=nuclear_norm_lmo,
                        tau=tau,
                        max_iter=self.cfg['max_iter'],
                        tol=self.cfg['tol'],
                        abs_tol=1e-6,
                        snapshot_interval=self.cfg['snapshot_interval'],
                        step_method=step,
                        fixed_step=self.cfg['fixed_step']
                    )
                    start = time.time()
                    solver.run()
                    duration = time.time() - start

                    n_iters = len(solver.weights_history)
                    print(f"Finished {solver_name}-{step} "
                          f"in {duration:.1f}s, iters={n_iters}")

                    # Reconstruct full X_k and compute RMSE
                    rmse_tr = np.zeros(n_iters)
                    rmse_te = np.zeros(n_iters)
                    for i, ws in enumerate(solver.weights_history):
                        Xk = sum(w * atom.to_matrix()
                                 for w, atom in zip(ws, solver.atoms))
                        Xk_full = Xk + mu_tr
                        rmse_tr[i] = evaluate(M_true, Xk_full, mask_train)
                        rmse_te[i] = evaluate(M_true, Xk_full, mask_test)

                    gaps     = np.array([h[1] for h in solver.history])
                    obj_vals = np.array([h[2] for h in solver.history])
                    times    = np.array(solver.times[1:n_iters+1])
                    steps_h  = np.array(solver.step_history)[:n_iters]
                    active_sz= np.array([len(ws) 
                                         for ws in solver.weights_history])

                    solver_map[step] = {
                        'snap_iters':   np.arange(1, n_iters+1),
                        'gap':          gaps,
                        'obj_vals':     obj_vals,
                        'rmse_train':   rmse_tr,
                        'rmse_test':    rmse_te,
                        'times':        times,
                        'step_history': steps_h,
                        'active_sizes': active_sz,
                    }

                    # summary metrics
                    final_tr, final_te = rmse_tr[-1], rmse_te[-1]
                    nrm_tr = final_tr / rating_range
                    nrm_te = final_te / rating_range
                    R2_tr = 1 - (final_tr**2 * mask_train.sum() / SST_tr)
                    R2_te = 1 - (final_te**2 * mask_test.sum()   / SST_te)
                    self.summary_rows.append(
                        (dataset, f"{solver_name}-{step}",
                         final_tr, final_te,
                         nrm_tr, nrm_te,
                         R2_tr,  R2_te,
                         n_iters)
                    )

                ds_results[solver_name] = solver_map   # ← correct key here

            self.results[dataset] = ds_results

        return self.results, self.summary_rows


class ExperimentPlotter:
    def __init__(self, results, summary_rows, cfg):
        self.results      = results
        self.summary_rows = summary_rows
        self.cfg          = cfg

    def _plot_gap(self, ax, data):
        for solver, dd in data.items():
            for step, d in dd.items():
                x = np.array(d['snap_iters'])
                y = np.array(d['gap'])
                L = min(len(x), len(y))
                x, y = x[:L], y[:L]
                idx = subsample_indices(L)
                ax.loglog(x[idx], np.maximum(y[idx], 1e-16), label=f"{solver}-{step}")
        ax.set(title='Gap vs Iter', xlabel='Iteration', ylabel='Duality Gap')
        ax.legend(); ax.grid(True, which='both')

    def _plot_obj_vs_time(self, ax, data):
        for solver, dd in data.items():
            for step, d in dd.items():
                x = np.array(d['times'])
                y = np.array(d['obj_vals'])
                L = min(len(x), len(y))
                x, y = x[:L], y[:L]
                idx = subsample_indices(L)
                ax.plot(x[idx], y[idx], label=f"{solver}-{step}")
        ax.set(title='Obj vs Time', xlabel='Time (s)', ylabel='Objective')
        ax.legend(); ax.grid(True)

    def _plot_rmse_vs_iter(self, ax, data):
        for solver, dd in data.items():
            for step, d in dd.items():
                x = np.array(d['snap_iters'])
                y_tr = np.array(d['rmse_train'])
                y_te = np.array(d['rmse_test'])
                L = min(len(x), len(y_tr), len(y_te))
                x, y_tr, y_te = x[:L], y_tr[:L], y_te[:L]
                idx = subsample_indices(L)
                ax.plot(x[idx], y_tr[idx], '--', label=f"{solver}-{step}-train")
                ax.plot(x[idx], y_te[idx],  '-', label=f"{solver}-{step}-test")
        ax.set(title='RMSE vs Iter', xlabel='Iteration', ylabel='RMSE')
        ax.legend(); ax.grid(True)

    def _plot_active_set(self, ax, data):
        pf = data.get('PFW', {})
        for step, d in pf.items():
            x = np.array(d['snap_iters'])
            y = np.array(d['active_sizes'])
            L = min(len(x), len(y))
            x, y = x[:L], y[:L]
            if L == 0: 
                continue
            idx = subsample_indices(L)
            ax.plot(x[idx], y[idx], '-o', label=step)
        ax.set(title='PFW Active-set Growth', xlabel='Iteration', ylabel='Active-set size')
        ax.legend(); ax.grid(True)

    def _plot_step_size(self, ax, data):
        for solver, dd in data.items():
            for step, d in dd.items():
                # step_history length = n_iters, snap_iters length = n_iters+1
                y = np.array(d['step_history'])
                x = np.array(d['snap_iters'])[1:len(y)+1]
                L = min(len(x), len(y))
                x, y = x[:L], y[:L]
                if L == 0:
                    continue
                idx = subsample_indices(L)
                ax.plot(x[idx], y[idx], 'o-', label=f"{solver}-{step}")
        ax.set(title='Step-size vs Iter', xlabel='Iteration', ylabel='Step-size γ')
        ax.set_yscale('log'); ax.legend(); ax.grid(True)

    def _plot_rmse_vs_rank(self, ax, data):
        for solver, dd in data.items():
            for step, d in dd.items():
                y = np.array(d['rmse_train'])
                if solver == 'PFW':
                    x_full = np.array(d['active_sizes'])
                else:
                    # for FW, rank = iteration number = snap_iters
                    x_full = np.array(d['snap_iters'])
                L = min(len(x_full), len(y))
                x, y = x_full[:L], y[:L]
                if L == 0:
                    continue
                idx = subsample_indices(L)
                ax.plot(x[idx], y[idx], '-o', label=f"{solver}-{step}")
        ax.set(title='Train RMSE vs Rank', xlabel='Rank', ylabel='Train RMSE')
        ax.legend(); ax.grid(True)

    def plot(self):
        for dataset, data in self.results.items():
            print(f"\n=== Results for {dataset} ===")
            fig, axes = plt.subplots(2, 3, figsize=(18, 10))
            for ax in axes.flatten():
                ax.set_axisbelow(True)

            self._plot_gap(axes[0,0], data)
            self._plot_obj_vs_time(axes[0,1], data)
            self._plot_rmse_vs_iter(axes[0,2], data)
            self._plot_active_set(axes[1,0], data)
            self._plot_step_size(axes[1,1], data)
            self._plot_rmse_vs_rank(axes[1,2], data)

            plt.tight_layout()
            plt.suptitle(f"Diagnostics {dataset}", y=1.02)
            if self.cfg['save_plots']:
                os.makedirs(self.cfg['plot_dir'], exist_ok=True)
                plt.savefig(os.path.join(self.cfg['plot_dir'], f"{dataset}_diagnostics.png"))
            plt.show()
            plt.close(fig)

        # Combined summary
        print("\n=== Combined Summary ===")
        hdr = (
            f"{'Dataset':10s}{'Solver-Step':20s}"
            f"{'RMSE_tr':>8s}{'RMSE_te':>8s}"
            f"{'NRMSE_tr':>8s}{'NRMSE_te':>8s}"
            f"{'R2_tr':>8s}{'R2_te':>8s}{'Iters':>6s}"
        )
        print(hdr)
        print('-'*len(hdr))
        for ds, sol, tr, te, nrtr, nrte, r2t, r2e, it in self.summary_rows:
            print(f"{ds:10s}{sol:20s}"
                  f"{tr:8.4f}{te:8.4f}"
                  f"{nrtr:8.4f}{nrte:8.4f}"
                  f"{r2t:8.4f}{r2e:8.4f}{it:6d}")



if __name__ == '__main__':
    runner = ExperimentRunner(config)
    results, summary_rows = runner.run()
    plotter = ExperimentPlotter(results, summary_rows, config)
    plotter.plot()
