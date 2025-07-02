import logging, time
import numpy as np
import matplotlib.pyplot as plt

from utils import load_dataset, evaluate
from solvers import (
    MatrixCompletionObjective,
    nuclear_norm_lmo,
    FrankWolfe,
    PairwiseFrankWolfe,
)

# ----------------------------------------------------------------------------
# Logging config
# ----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s: %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")

# ----------------------------------------------------------------------------
# Experiment config
# ----------------------------------------------------------------------------
config = {
    'dataset':       'ml-100k',
    'test_fraction': 0.2,
    'seed':          42,
    'tau_scale':     1.0,
    'max_iter':      100,
    'tol':           1e-6,
    'tol_obj':       1e-8,
    'patience':      10,
}
STEP_METHODS = ['analytic']

# ----------------------------------------------------------------------------
# Load data
# ----------------------------------------------------------------------------
logging.info(f"Loading {config['dataset']}")
M_obs, mask_train, M_true = load_dataset(
    name=config['dataset'],
    test_fraction=config['test_fraction'],
    seed=config['seed']
)
obj = MatrixCompletionObjective(M_obs, mask_train)
tau = config['tau_scale'] * np.linalg.norm(M_true, ord='nuc')
logging.info(f"tau = {tau:.3f}")

# ----------------------------------------------------------------------------
# Run experiments
# ----------------------------------------------------------------------------
results = {}
solver_objs = {}
for name, Solver in [('FW', FrankWolfe), ('PFW', PairwiseFrankWolfe)]:
    results[name] = {}
    solver_objs[name] = {}
    for step in STEP_METHODS:
        logging.info(f"→ {name} with step={step}")
        solver = Solver(
            objective=obj,
            lmo_fn=nuclear_norm_lmo,
            tau=tau,
            max_iter=config['max_iter'],
            tol=config['tol'],
            step_method=step,
            tol_obj=config['tol_obj'],
            patience=config['patience']
        )
        t0 = time.time()
        Xf = solver.run()
        dt = time.time() - t0
        logging.info(f"   done in {dt:.2f}s, iters={len(solver.history)}")

        # collect histories
        n = len(solver.history)
        if n == 0:
            continue
        iters    = np.array([h[0] for h in solver.history])
        gaps     = np.array([h[1] for h in solver.history])
        objs     = np.array([h[2] for h in solver.history])
        rmse_tr  = np.sqrt(2*objs / mask_train.sum())
        # align test rmse with history length
        snapshots = solver.snapshots[:n]
        rmse_te  = np.array([evaluate(M_true, Xk, ~mask_train)
                             for Xk in snapshots])
        # align times with history length
        times    = np.array(solver.times)
        times    = times[1:n+1] if len(times) >= n+1 else times[:n]

        results[name][step] = {
            'iters': iters,
            'gap': gaps,
            'obj_vals': objs,
            'rmse_train': rmse_tr,
            'rmse_test': rmse_te,
            'times': times,
            'step_history': np.array(solver.step_history),
            'active_sizes': np.array([
                len(w) for w in getattr(solver, 'weights_history', [])
            ])[:n]
        }
        solver_objs[name][step] = solver

# ----------------------------------------------------------------------------
# Plot 1: Dual gap vs iteration
# ----------------------------------------------------------------------------
plt.figure(figsize=(6,4))
for name, data_dict in results.items():
    for step, data in data_dict.items():
        g = np.maximum(data['gap'],1e-16)
        plt.loglog(data['iters'], g, label=f"{name}-{step}")
plt.xlabel("Iteration")
plt.ylabel("Duality Gap")
plt.title("Duality Gap vs Iteration")
plt.legend()
plt.grid(True, which='both')
plt.tight_layout()

# ----------------------------------------------------------------------------
# Plot 2: Objective value vs time
# ----------------------------------------------------------------------------
plt.figure(figsize=(6,4))
for name, data_dict in results.items():
    for step, data in data_dict.items():
        plt.plot(data['times'], data['obj_vals'], label=f"{name}-{step}")
plt.xlabel("Time (s)")
plt.ylabel("Objective value")
plt.title("Objective vs Wall-clock Time")
plt.legend()
plt.grid(True)
plt.tight_layout()

# ----------------------------------------------------------------------------
# Plot 3: RMSE train/test vs iteration
# ----------------------------------------------------------------------------
plt.figure(figsize=(6,4))
for name, data_dict in results.items():
    for step, data in data_dict.items():
        it = data['iters']
        plt.plot(it, data['rmse_train'], '--', label=f"{name}-{step}-train")
        plt.plot(it, data['rmse_test'],  '-', label=f"{name}-{step}-test")
plt.xlabel("Iteration")
plt.ylabel("RMSE")
plt.title("Train/Test RMSE vs Iteration")
plt.legend()
plt.grid(True)
plt.tight_layout()

# ----------------------------------------------------------------------------
# Plot 4: Active-set size vs iteration (PFW only)
# ----------------------------------------------------------------------------
plt.figure(figsize=(6,3))
for step, data in results.get('PFW', {}).items():
    plt.plot(data['iters'], data['active_sizes'], '-o', label=step)
plt.xlabel("Iteration")
plt.ylabel("Active-set size")
plt.title("PFW Active-set Growth")
plt.legend()
plt.grid(True)
plt.tight_layout()

# ----------------------------------------------------------------------------
# Plot 5: Step-size vs iteration
# ----------------------------------------------------------------------------
plt.figure(figsize=(6,4))
for name, data_dict in results.items():
    for step, data in data_dict.items():
        steps = data.get('step_history', [])
        if len(steps) == 0:
            continue
        its = np.arange(len(steps))
        plt.plot(its, steps, label=f"{name}-{step}")
plt.xlabel("Iteration")
plt.ylabel("Step-size γₖ")
plt.yscale('log')
plt.title("Step-size vs Iteration")
plt.legend()
plt.grid(True)
plt.tight_layout()

# ----------------------------------------------------------------------------
# Plot 6: RMSE vs estimated rank
# ----------------------------------------------------------------------------
plt.figure(figsize=(6,4))
for name, solvers_map in solver_objs.items():
    for step, solver in solvers_map.items():
        ranks = [np.linalg.matrix_rank(Xk, tol=1e-3) for Xk in solver.snapshots[:len(solver.history)]]
        plt.plot(ranks, results[name][step]['rmse_train'], '-o', label=f"{name}-{step}")
plt.xlabel("Estimated Rank")
plt.ylabel("Train RMSE")
plt.title("Train RMSE vs Estimated Rank")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.show()

# ----------------------------------------------------------------------------
# Summary table
# ----------------------------------------------------------------------------
print("\nFinal Metrics:")
print(f"{'Solver-Step':20s} {'Train RMSE':>10s} {'Test RMSE':>10s} {'Iters':>6s} {'Time(s)':>8s}")
for name, data_dict in results.items():
    for step, data in data_dict.items():
        trm = data['rmse_train'][-1]
        trem= data['rmse_test'][-1]
        it  = len(data['iters'])
        ti  = data['times'][-1] if len(data['times'])>0 else 0.0
        print(f"{name+'-'+step:20s} {trm:10.4f} {trem:10.4f} {it:6d} {ti:8.2f}")
