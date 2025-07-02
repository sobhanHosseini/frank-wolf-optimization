import numpy as np
import pandas as pd

def evaluate(M_true, X_pred, mask):
    """
    Compute RMSE on entries where mask is True.
    """
    diff = (X_pred - M_true)[mask]
    if diff.size == 0:
        return 0.0
    return np.sqrt(np.mean(diff ** 2))

def load_jester2(path="./data/jester2/jester_ratings.dat",
                 test_fraction=0.2, seed=0):
    import numpy as np

    # 1) First pass: build maps for contiguous indexing
    user_map = {}
    item_map = {}
    reviews  = []
    with open(path, 'r') as f:
        for line in f:
            u, i, r = line.split()
            if u not in user_map:
                user_map[u] = len(user_map)
            if i not in item_map:
                item_map[i] = len(item_map)
            reviews.append((u, i, float(r)))

    n_users  = len(user_map)   # should be 59132
    n_items  = len(item_map)   # should be 150

    # 2) Build the full matrix
    M_true = np.zeros((n_users, n_items), dtype=float)
    for u, i, r in reviews:
        ui = user_map[u]
        ii = item_map[i]
        M_true[ui, ii] = r

    # 3) Train/test split
    rng  = np.random.RandomState(seed)
    mask = rng.rand(n_users, n_items) < (1 - test_fraction)
    M_obs = np.zeros_like(M_true)
    M_obs[mask] = M_true[mask]

    return M_obs, mask, M_true


def load_movielens100k(path="./data/ml-100k/u.data",
                       test_fraction=0.2,
                       seed=0):
    """
    Load MovieLens-100K from u.data, then do an 80/20 train/test split.
    Each row in u.data: userID   itemID   rating   timestamp
    """
    df = pd.read_csv(path, sep="\t", names=["user","item","rating","ts"])
    n_u = df.user.max()
    n_i = df.item.max()

    # Build full rating matrix
    M_true = np.zeros((n_u, n_i), dtype=float)
    for u,i,r in zip(df.user, df.item, df.rating):
        M_true[u-1, i-1] = r

    # Random train/test split
    rng  = np.random.RandomState(seed)
    mask = rng.rand(n_u, n_i) < (1 - test_fraction)
    M_obs = np.zeros_like(M_true)
    M_obs[mask] = M_true[mask]

    return M_obs, mask, M_true


def load_movielens1m(
    path="./data/ml-1m/ratings.dat",
    test_fraction=0.2,
    seed=0
):
    """
    Load MovieLens-1M:
      ratings.dat lines: UserID::MovieID::Rating::Timestamp
    Returns (M_obs, mask_train, M_true).
    """
    # 1) Read all ratings
    df = pd.read_csv(
        path, sep="::", engine="python",
        names=["user", "item", "rating", "_"]
    )
    n_users = df.user.max()
    n_items = df.item.max()

    # 2) Build the full User×Item matrix
    M_true = np.zeros((n_users, n_items), dtype=float)
    for u, i, r in zip(df.user, df.item, df.rating):
        M_true[u - 1, i - 1] = r

    # 3) Random 80/20 train/test split
    rng  = np.random.RandomState(seed)
    mask = rng.rand(n_users, n_items) < (1 - test_fraction)
    M_obs = np.zeros_like(M_true)
    M_obs[mask] = M_true[mask]

    return M_obs, mask, M_true


def load_dataset(name, **kwargs):
    """
    Dispatch to the appropriate loader.

    Parameters
    ----------
    name : str
        One of 'ml-100k', 'jester2', 'ml-1m'.
    **kwargs : dict
        Passed through to the specific loader (e.g. path, test_fraction, seed).

    Returns
    -------
    M_obs, mask_train, M_true
    """
    if name == "ml-100k":
        return load_movielens100k(**kwargs)
    elif name == "jester2":
        return load_jester2(**kwargs)
    elif name == "ml-1m":
        return load_movielens1m(**kwargs)
    else:
        raise ValueError(f"Unknown dataset '{name}'. "
                         "Choose from 'ml-100k', 'jester2', 'ml-1m'.")
