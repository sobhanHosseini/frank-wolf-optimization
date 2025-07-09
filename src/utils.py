# utils.py

import numpy as np
import pandas as pd
from scipy import sparse

def approximate_nuclear_norm(M: sparse.csr_matrix, k: int = 10,
                             tol: float = 1e-3, maxiter: int = 200) -> float:
    try:
        u, s, vt = sparse.linalg.svds(M, k=min(k, min(M.shape)-1),
                                      tol=tol, maxiter=maxiter)
        return float(np.sum(s))
    except Exception:
        return float(np.linalg.norm(M.toarray(), ord='nuc'))

def train_test_split_matrix(M_true: np.ndarray, test_fraction: float = 0.2,
                            seed: int = 0):
    rng    = np.random.RandomState(seed)
    rows, cols = np.nonzero(M_true)
    train_flag = rng.rand(rows.size) < (1 - test_fraction)

    mask_train = np.zeros_like(M_true, dtype=bool)
    mask_test  = np.zeros_like(M_true, dtype=bool)
    mask_train[rows[train_flag], cols[train_flag]] = True
    mask_test[rows[~train_flag], cols[~train_flag]]  = True

    data     = M_true[mask_train]
    obs_rows, obs_cols = np.where(mask_train)
    M_obs    = sparse.csr_matrix((data, (obs_rows, obs_cols)), shape=M_true.shape)
    return M_obs, mask_train, mask_test, M_true

def load_jester2(path="../data/jester2/jester_ratings.dat",
                 test_fraction=0.2, seed=0):
    user_map, item_map, reviews = {}, {}, []
    with open(path, 'r') as f:
        for line in f:
            u, i, r = line.split()
            user_map.setdefault(u, len(user_map))
            item_map.setdefault(i, len(item_map))
            reviews.append((u, i, float(r)))
    n_users, n_items = len(user_map), len(item_map)
    M_true = np.zeros((n_users, n_items), dtype=float)
    for u, i, r in reviews:
        M_true[user_map[u], item_map[i]] = r
    return train_test_split_matrix(M_true, test_fraction, seed)

def load_movielens100k(path="../data/ml-100k/u.data",
                       test_fraction=0.2, seed=0):
    df = pd.read_csv(path, sep="\t", names=["user","item","rating","ts"])
    n_u, n_i = df.user.max(), df.item.max()
    M_true = np.zeros((n_u, n_i), dtype=float)
    for u,i,r in zip(df.user, df.item, df.rating):
        M_true[u-1, i-1] = r
    return train_test_split_matrix(M_true, test_fraction, seed)

def load_movielens1m(path="../data/ml-1m/ratings.dat",
                     test_fraction=0.2, seed=0):
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

def evaluate(M_true, X_pred, mask):
    diff = (X_pred - M_true)[mask]
    return np.sqrt(np.mean(diff**2)) if diff.size else 0.0
