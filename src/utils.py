import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds

# ----------------------------------------------------------------
# Helper: approximate nuclear norm for faster tau estimation
# ----------------------------------------------------------------
def approximate_nuclear_norm(M: np.ndarray, k: int = 10, tol: float = 1e-3, maxiter: int = 200) -> float:
    """
    Approximate the nuclear norm (trace norm) of M by summing its top-k singular values via svds.
    """
    # ensure k is valid
    k = min(k, min(M.shape) - 1)
    if k <= 0:
        return np.linalg.norm(M, ord='nuc')
    try:
        u, s, vt = svds(M, k=k, tol=tol, maxiter=maxiter)
        return float(np.sum(s))
    except Exception:
        # fallback to exact
        return float(np.linalg.norm(M, ord='nuc'))

# ----------------------------------------------------------------
# Train/test split utilities (only among observed entries)
# ----------------------------------------------------------------
def train_test_split_matrix(M_true: np.ndarray, test_fraction: float = 0.2, seed: int = 0):
    """
    Splits only over non-zero entries of M_true into train/test masks.
    Returns: M_obs, mask_train, mask_test
    """
    rng = np.random.RandomState(seed)
    # find indices of observed entries
    ui, ij = np.nonzero(M_true)
    n_obs = ui.size
    # sample train/test
    train_flag = rng.rand(n_obs) < (1 - test_fraction)
    mask_train = np.zeros_like(M_true, dtype=bool)
    mask_train[ui[train_flag], ij[train_flag]] = True
    mask_test = np.zeros_like(M_true, dtype=bool)
    mask_test[ui[~train_flag], ij[~train_flag]] = True
    # observed matrix
    M_obs = np.zeros_like(M_true)
    M_obs[mask_train] = M_true[mask_train]
    return M_obs, mask_train, mask_test

# ----------------------------------------------------------------
# Loaders
# ----------------------------------------------------------------
def load_jester2(path="./data/jester2/jester_ratings.dat",
                 test_fraction=0.2, seed=0):
    # read ratings
    user_map = {}
    item_map = {}
    reviews  = []
    with open(path, 'r') as f:
        for line in f:
            u, i, r = line.split()
            user_map.setdefault(u, len(user_map))
            item_map.setdefault(i, len(item_map))
            reviews.append((u, i, float(r)))
    n_users = len(user_map)
    n_items = len(item_map)
    # build full
    M_true = np.zeros((n_users, n_items), dtype=float)
    for u, i, r in reviews:
        M_true[user_map[u], item_map[i]] = r
    return train_test_split_matrix(M_true, test_fraction, seed)


def load_movielens100k(path="./data/ml-100k/u.data",
                       test_fraction=0.2,
                       seed=0):
    df = pd.read_csv(path, sep="\t", names=["user","item","rating","ts"])
    n_u, n_i = df.user.max(), df.item.max()
    M_true = np.zeros((n_u, n_i), dtype=float)
    for u,i,r in zip(df.user, df.item, df.rating):
        M_true[u-1, i-1] = r
    return train_test_split_matrix(M_true, test_fraction, seed)


def load_movielens1m(path="./data/ml-1m/ratings.dat",
                     test_fraction=0.2,
                     seed=0):
    df = pd.read_csv(path, sep="::", engine="python",
                     names=["user","item","rating","_"])
    n_u, n_i = df.user.max(), df.item.max()
    M_true = np.zeros((n_u, n_i), dtype=float)
    for u,i,r in zip(df.user, df.item, df.rating):
        M_true[u-1, i-1] = r
    return train_test_split_matrix(M_true, test_fraction, seed)


def load_dataset(name, **kwargs):
    if name == "ml-100k":
        return load_movielens100k(**kwargs)
    elif name == "jester2":
        return load_jester2(**kwargs)
    elif name == "ml-1m":
        return load_movielens1m(**kwargs)
    else:
        raise ValueError(f"Unknown dataset '{name}'")

# ----------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------
def evaluate(M_true, X_pred, mask):
    diff = (X_pred - M_true)[mask]
    return np.sqrt(np.mean(diff**2)) if diff.size else 0.0
