# Notebook-friendly main.py with early-stopping config
import time
import numpy as np
import matplotlib.pyplot as plt

from utils import load_dataset, evaluate
from solvers import (
    MatrixCompletionObjective,
    nuclear_norm_lmo,
    FrankWolfe,
    PairwiseFrankWolfe,
)

# ---------------------
# Configuration
# ---------------------
config = {
    'dataset': 'ml-100k',         # options: 'ml-100k', 'jester2', 'ml-1m'
    'path': None,                 # custom file path, or None for default
    'test_fraction': 0.2,
    'seed': 42,
    'tau': None,                  # None -> defaults to ||M_true||_*
    'max_iter': 200,
    'tol': 1e-4,                  # duality-gap tolerance
    'tol_obj': 1e-6,              # objective improvement tolerance
    'patience': 10,               # early-stopping patience
    'step_method': 'vanilla',     # options: 'vanilla','analytic','line_search','armijo'
    'solvers': ['fw', 'pfw'],     # choose subset of ['fw','pfw']
}

# ---------------------
# 1) Load Data
# ---------------------
print(f"Loading dataset '{config['dataset']}'...")
load_kwargs = {
    'name': config['dataset'], 
    'test_fraction': config['test_fraction'], 
    'seed': config['seed']
}
if config['path']:
    load_kwargs['path'] = config['path']
M_obs, mask_train, M_true = load_dataset(**load_kwargs)
print("Done.")

# ---------------------
# 2) Setup Objective & LMO
# ---------------------
obj = MatrixCompletionObjective(M_obs, mask_train)
def_tau = np.linalg.norm(M_true, ord='nuc')
tau = config['tau'] if config['tau'] is not None else def_tau
print(f"Using tau = {tau:.3f} (default was {def_tau:.3f})")

# ---------------------
# 3) Run Solvers
# ---------------------
results = {}

for solver_key in config['solvers']:
    if solver_key == 'fw':
        solver_cls = FrankWolfe;
        name = 'FW'
    elif solver_key == 'pfw':
        solver_cls = PairwiseFrankWolfe;
        name = 'PFW'
    else:
        continue

    print(f"\nRunning {name}... ")
    solver = solver_cls(
        objective=obj,
        lmo_fn=nuclear_norm_lmo,
        tau=tau,
        max_iter=config['max_iter'],
        tol=config['tol'],
        step_method=config['step_method'],
        tol_obj=config['tol_obj'],
        patience=config['patience']
    )
    t0 = time.time()
    X = solver.run()
    duration = time.time() - t0
    iters, gaps, vals = zip(*solver.history)
    iters = np.array(iters)
    gaps = np.array(gaps)
    vals = np.array(vals)

    rmse_train = np.sqrt(2 * obj.value(X) / mask_train.sum())
    rmse_test  = evaluate(M_true, X, ~mask_train)

    print(f"{name} finished in {duration:.2f}s "
          f"(iters={len(iters)}): \n"
          f"  Train RMSE = {rmse_train:.4f}, Test RMSE = {rmse_test:.4f}")

    results[name] = (iters, gaps)

# ---------------------
# 4) Plot Results
# ---------------------
plt.figure(figsize=(6,4))
for name, (iters, gaps) in results.items():
    plt.loglog(iters, gaps, label=name)
plt.xlabel('Iteration')
plt.ylabel('Duality Gap')
plt.title(f"{config['dataset']}: FW vs Pairwise-FW (step={config['step_method']})")
plt.legend()
plt.tight_layout()
plt.show()
