# Projection-Free Matrix Completion with Frank–Wolfe

A modular Python codebase for projection-free matrix completion using Frank–Wolfe (FW) and Pairwise Frank–Wolfe (PFW) algorithms.

## 🚀 Features

* **Step-size options:** analytic line-search, vanilla diminishing (`2/(t+2)`), fixed.
* **Projection-free nuclear-norm LMO:** warm-started power-method implementation.
* **Active-set tracking:** maintain rank-1 atoms and weight updates with pruning in PFW.
* **Benchmark datasets:** MovieLens-100k, MovieLens-1M, Jester2 with 80/20 splits.
* **Diagnostics & plots:**

  * Duality gap vs. iteration
  * Training objective vs. time
  * Test RMSE vs. iteration
  * Solve vs. evaluation time breakdown

---

## 💾 Repository Structure

```
project-root/
├── data/                 # Raw benchmark files
│   ├── ml-100k/
│   │   └── u.data
│   ├── ml-1m/
│   │   └── ratings.dat
│   └── jester2/
│       └── jester_ratings.dat
└── src/                  # Source code
    ├── main.py           # Experiment runner & plotting
    ├── solvers.py        # LMO, FrankWolfe, PairwiseFrankWolfe
    └── utils.py          # Data loading, nuclear-norm approx, RMSE
```

## ⚙️ Configuration

Edit the `config` dictionary in `src/main.py` to select datasets, step-size rules, tolerance, and plotting options. Example:

```python
config = {
  'datasets':           ['ml-100k', 'jester2', 'ml-1m'],
  'steps':              ['analytic', 'vanilla'],
  'test_fraction':      0.2,
  'seed':               42,
  'tau_scale':          1.0,
  'tau_approx_k':       12,
  'max_iter':           100,
  'tol':                1e-2,
  'abs_tol':            1e-6,
  'fixed_step':         0.1,
  'snapshot_interval':  10,
  'save_plots':         True,
  'plot_dir':           'plots'
}
```

## ▶️ Running Experiments

From project root execute:

```bash
python -m src.main
```

This will:

* Load each dataset and split into train/test
* Center ratings by subtracting the training mean
* Approximate nuclear-norm bound τ via truncated SVD
* Run FW/PFW variants per configuration
* Record timings, duality gap, objective, and RMSE
* Save plots under `plots/`

## 📝 Code Overview

### `src/utils.py`

* **`load_dataset`**: loads MovieLens and Jester2, returns `(M_obs, mask_train, mask_test, M_true)`
* **`approximate_nuclear_norm`**: truncated SVD-based estimation of ‖M‖\_\*
* **`evaluate`**: computes RMSE on masked entries

### `src/solvers.py`

* **`power_method`**: warm-started power iteration for top singular vector
* **`nuclear_norm_lmo`**: builds rank-1 atom for the LMO
* **`FrankWolfe`**: implements classic FW with various step-size rules
* **`PairwiseFrankWolfe`**: extends FW with away-step updates and pruning

### `src/main.py`

* **`ExperimentRunner`**:

  1. Instantiates solvers for each (method, step)
  2. Runs `.run()`, measures solve & eval times
  3. Reconstructs predictions from atoms + weights, computes RMSE
  4. Aggregates diagnostics into `summary_rows`
  5. Generates convergence and time-breakdown plots
