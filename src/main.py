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
# Configuration: datasets and step rules
# ----------------------------------------------------------------------------
config = {
    'datasets':       ['ml-100k', 'jester2'],   # list of datasets
    'steps':          ['analytic'],              # step-size methods
    'test_fraction':  0.2,
    'seed':           42,
    'tau_scale':      1.0,
    'max_iter':       30,
    'tol':            1e-6,
    'tol_obj':        1e-8,
    'patience':       10,
    'save_plots':     False,
    'plot_dir':       'plots',
    'tau_approx_k':   10,  # top-k for approximate nuclear norm
    'snapshot_interval': 5,  # interval for saving snapshots
}
if config['save_plots']:
    os.makedirs(config['plot_dir'], exist_ok=True)

# ----------------------------------------------------------------------------
# Utility: subsample indices for plotting
# ----------------------------------------------------------------------------
def subsample_indices(n, num=20):
    if n <= num:
        return np.arange(n)
    return np.unique(np.linspace(0, n - 1, num, dtype=int))

# ----------------------------------------------------------------------------
# Experiment runner: executes FW and PFW, collects metrics
# ----------------------------------------------------------------------------
class ExperimentRunner:
    def __init__(self, cfg):
        self.cfg = cfg
        self.results = {}
        self.summary_rows = []

    def run(self):
        for dataset in self.cfg['datasets']:
            print(f"\n=== Dataset: {dataset} ===")
            # load data: M_obs, mask_train, mask_test, M_true
            M_obs, mask_train, mask_test, M_true = load_dataset(
                name=dataset,
                test_fraction=self.cfg['test_fraction'],
                seed=self.cfg['seed']
            )

            # center by train-mean
            mu_tr = M_true[mask_train].mean()
            M_obs_centered = M_obs.copy()
            M_obs_centered[mask_train] -= mu_tr

            # rating range for normalization
            if dataset == 'jester2':
                rmin, rmax = -10.0, 10.0
            elif dataset in ('ml-100k', 'ml-1m'):
                rmin, rmax = 1.0, 5.0
            else:
                vals = M_true[mask_train]
                rmin, rmax = float(vals.min()), float(vals.max())
            rating_range = rmax - rmin

            # centered train/test vectors
            y_tr = M_true[mask_train] - mu_tr
            y_te = M_true[mask_test]  - mu_tr
            SST_tr = (y_tr**2).sum()
            SST_te = (y_te**2).sum()

            # objective and tau
            obj = MatrixCompletionObjective(M_obs_centered, mask_train)
            def_tau = approximate_nuclear_norm(M_obs_centered, k=self.cfg['tau_approx_k'])
            tau = self.cfg['tau_scale'] * def_tau
            print(f"μ_train={mu_tr:.3f}, τ≈{def_tau:.3f} → using τ={tau:.3f}")

            ds_results = {}
            for solver_name, Solver in [('FW', FrankWolfe), ('PFW', PairwiseFrankWolfe)]:
                solver_map = {}
                for step in self.cfg['steps']:
                    print(f"--- {solver_name}-{step} ---")
                    solver = Solver(
                        objective=obj,
                        lmo_fn=nuclear_norm_lmo,
                        tau=tau,
                        max_iter=self.cfg['max_iter'],
                        tol=self.cfg['tol'],
                        step_method=step,
                        tol_obj=self.cfg['tol_obj'],
                        patience=self.cfg['patience'],
                        snapshot_interval=self.cfg['snapshot_interval']
                    )
                    start = time.time()
                    solver.run()
                    duration = time.time() - start
                    print(f"Finished {solver_name}-{step} in {duration:.1f}s, iters={len(solver.history)}")

                    snaps = solver.snapshots
                    snap_iters = np.array(solver.snapshot_iters[:len(snaps)])
                    # compute RMSE per snapshot
                    rmse_tr = np.array([
                        np.sqrt(np.mean(((Xk + mu_tr)[mask_train] - M_true[mask_train])**2))
                        for Xk in snaps
                    ])
                    rmse_te = np.array([
                        evaluate(M_true, Xk + mu_tr, mask_test)
                        for Xk in snaps
                    ])
                    gaps = np.array([h[1] for h in solver.history])[:len(snaps)]
                    obj_vals = np.array([h[2] for h in solver.history])[:len(snaps)]
                    times = np.array(solver.times)[:len(snaps)]
                    steps_hist = np.array(solver.step_history)[:len(snaps)]
                    weights_hist = getattr(solver, 'weights_history', None)
                    active_sizes = (
                        np.array([len(w) for w in weights_hist])[:len(snaps)]
                        if weights_hist else np.zeros(len(snaps), int)
                    )

                    solver_map[step] = {
                        'snap_iters':   snap_iters,
                        'gap':          gaps,
                        'obj_vals':     obj_vals,
                        'rmse_train':   rmse_tr,
                        'rmse_test':    rmse_te,
                        'times':        times,
                        'step_history': steps_hist,
                        'active_sizes': active_sizes,
                    }

                    # summary
                    final_tr, final_te = rmse_tr[-1], rmse_te[-1]
                    nrm_tr = final_tr / rating_range
                    nrm_te = final_te / rating_range
                    R2_tr = 1 - ((final_tr**2 * mask_train.sum()) / SST_tr)
                    R2_te = 1 - ((final_te**2 * mask_test.sum())  / SST_te)
                    self.summary_rows.append(
                        (dataset, f"{solver_name}-{step}",
                         final_tr, final_te,
                         nrm_tr, nrm_te,
                         R2_tr,  R2_te,
                         len(snaps))
                    )

                ds_results[solver_name] = solver_map
            self.results[dataset] = ds_results

        return self.results, self.summary_rows

# ----------------------------------------------------------------------------
# Plotter: uses snap_iters to align RMSE curves
# ----------------------------------------------------------------------------
class ExperimentPlotter:
    def __init__(self, results, summary_rows, cfg):
        self.results      = results
        self.summary_rows = summary_rows
        self.cfg          = cfg

    def _plot_gap(self, ax, data):
        for sname, dd in data.items():
            for step, d in dd.items():
                idx = subsample_indices(len(d['snap_iters']))
                ax.loglog(
                    d['snap_iters'][idx],
                    np.maximum(d['gap'][idx], 1e-16),
                    label=f"{sname}-{step}"
                )
        ax.set(title='Gap vs Iter', xlabel='Iteration', ylabel='Duality Gap')
        ax.legend(); ax.grid(True, which='both')

    def _plot_obj_vs_time(self, ax, data):
        for sname, dd in data.items():
            for step, d in dd.items():
                idx = subsample_indices(len(d['times']))
                ax.plot(
                    d['times'][idx],
                    d['obj_vals'][idx],
                    label=f"{sname}-{step}"
                )
        ax.set(title='Obj vs Time', xlabel='Time (s)', ylabel='Objective')
        ax.legend(); ax.grid(True)

    def _plot_rmse_vs_iter(self, ax, data):
        for sname, dd in data.items():
            for step, d in dd.items():
                idx = subsample_indices(len(d['snap_iters']))
                ax.plot(
                    d['snap_iters'][idx],
                    d['rmse_train'][idx],
                    '--',
                    label=f"{sname}-{step}-train"
                )
                ax.plot(
                    d['snap_iters'][idx],
                    d['rmse_test'][idx],
                    '-',
                    label=f"{sname}-{step}-test"
                )
        ax.set(title='RMSE vs Iter', xlabel='Iteration', ylabel='RMSE')
        ax.legend(); ax.grid(True)

    def _plot_active_set(self, ax, data):
        if 'PFW' in data:
            for step, d in data['PFW'].items():
                idx = subsample_indices(len(d['snap_iters']))
                ax.plot(
                    d['snap_iters'][idx],
                    d['active_sizes'][idx],
                    '-o',
                    label=step
                )
        ax.set(title='PFW Active-set Growth', xlabel='Iteration', ylabel='Active-set size')
        ax.legend(); ax.grid(True)

    def _plot_step_size(self, ax, data):
        for sname, dd in data.items():
            for step, d in dd.items():
                idx = subsample_indices(len(d['snap_iters']))
                ax.plot(
                    d['snap_iters'][idx],
                    d['step_history'][idx],
                    'o-',
                    label=f"{sname}-{step}"
                )
        ax.set(title='Step-size vs Iter', xlabel='Iteration', ylabel='Step-size γₖ')
        ax.set_yscale('log'); ax.legend(); ax.grid(True)

    def _plot_rmse_vs_rank(self, ax, data):
        for sname, dd in data.items():
            for step, d in dd.items():
                # for FW: rank = iter_idx + 1; for PFW: rank = active_sizes
                if sname == 'PFW':
                    ranks = d['active_sizes']
                else:
                    ranks = d['snap_iters']  # each FW adds one rank per iter
                idx = subsample_indices(len(ranks))
                ax.plot(
                    np.array(ranks)[idx],
                    d['rmse_train'][idx],
                    '-o',
                    label=f"{sname}-{step}"
                )
        ax.set(title='Train RMSE vs Rank', xlabel='Rank', ylabel='Train RMSE')
        ax.legend(); ax.grid(True)

    def plot(self):
        for dataset, data in self.results.items():
            print(f"\n=== Results for {dataset} ===")
            fig, axes = plt.subplots(2, 3, figsize=(18, 10))
            for ax in axes.flatten():
                ax.set_axisbelow(True)

            # call each subplot
            self._plot_gap(axes[0,0], data)
            self._plot_obj_vs_time(axes[0,1], data)
            self._plot_rmse_vs_iter(axes[0,2], data)
            self._plot_active_set(axes[1,0], data)
            self._plot_step_size(axes[1,1], data)
            self._plot_rmse_vs_rank(axes[1,2], data)

            plt.tight_layout()
            plt.suptitle(f"Diagnostics {dataset}", y=1.02)
            if self.cfg['save_plots']:
                fig.savefig(os.path.join(self.cfg['plot_dir'], f"{dataset}_diagnostics.png"))
            plt.show()
            plt.close(fig)

            # Summary table per dataset
            rows = [r for r in self.summary_rows if r[0] == dataset]
            hdr = (
                f"{'Dataset':10s}{'Solver-Step':20s}{'RMSE_tr':>8s}"
                f"{'RMSE_te':>8s}{'NRMSE_tr':>8s}{'NRMSE_te':>8s}"
                f"{'R2_tr':>8s}{'R2_te':>8s}{'Iters':>6s}"
            )
            print(hdr)
            print('-' * len(hdr))
            for ds, solstep, tr, te, ntr, nte, r2t, r2e, it in rows:
                print(f"{ds:10s}{solstep:20s}{tr:8.4f}{te:8.4f}{ntr:8.4f}{nte:8.4f}{r2t:8.4f}{r2e:8.4f}{it:6d}")

        # Combined table across all datasets
        if len(self.cfg['datasets'])>1:
            print("\n=== Combined Summary ===")
            hdr = f"{'Dataset':10s}{'Solver-Step':20s}{'RMSE_tr':>8s}{'RMSE_te':>8s}{'NRMSE_tr':>8s}{'NRMSE_te':>8s}{'R2_tr':>8s}{'R2_te':>8s}{'Iters':>6s}"
            print(hdr)
            print('-'*len(hdr))
            for ds, solstep, tr, te, ntr, nte, r2t, r2e, it in self.summary_rows:
                print(f"{ds:10s}{solstep:20s}{tr:8.4f}{te:8.4f}{ntr:8.4f}{nte:8.4f}{r2t:8.4f}{r2e:8.4f}{it:6d}")

        # Cross-dataset NRMSE plot
        if len(self.cfg['datasets'])>1:
            ds_list = self.cfg['datasets']
            x = np.arange(len(ds_list))
            fig, ax = plt.subplots(figsize=(6,4))
            for solver_name in ['FW','PFW']:
                for step in self.cfg['steps']:
                    y = []
                    for ds in ds_list:
                        d = self.results[ds][solver_name][step]
                        y.append(d['rmse_test'][-1] / (20 if ds=='jester2' else 4))
                    ax.plot(x, y, '-o', label=f"{solver_name}-{step}")
            ax.set_xticks(x); ax.set_xticklabels(ds_list)
            ax.set(title='Test NRMSE Across Datasets', xlabel='Dataset', ylabel='Test NRMSE')
            ax.legend(); ax.grid(True)
            plt.tight_layout()
            if self.cfg['save_plots']:
                fig.savefig(os.path.join(self.cfg['plot_dir'], "cross_dataset_nrmse.png"))
            plt.show()

# ----------------------------------------------------------------------------
# Main execution
# ----------------------------------------------------------------------------
if __name__ == '__main__':
    runner = ExperimentRunner(config)
    results, summary_rows = runner.run()
    plotter = ExperimentPlotter(results, summary_rows, config)
    plotter.plot()
