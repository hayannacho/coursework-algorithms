import numpy as np


def my_recommender(rate_mat, lr, with_reg, seed=42):
    rng = np.random.RandomState(seed)
    rate_mat = np.asarray(rate_mat, dtype=float)
    if (
        rate_mat.ndim != 2
        or not np.isfinite(rate_mat).all()
        or not (rate_mat > 0).any()
        or (rate_mat < 0).any()
    ):
        raise ValueError("Expected finite nonnegative matrix with observed positive ratings")
    if not isinstance(lr, int) or lr < 1:
        raise ValueError("rank must be positive")
    n_user, n_item = rate_mat.shape
    rate_mat = rate_mat.astype(np.float64)
    U = rng.rand(n_user, lr) / lr
    V = rng.rand(n_item, lr) / lr
    b_u = np.zeros((n_user, 1))
    b_i = np.zeros((n_item, 1))
    global_bias = np.sum(rate_mat) / np.sum(rate_mat > 0)
    max_iter = 30
    learning_rate = 0.01
    if with_reg:
        u_reg = 0.02
        v_reg = 0.02
        b_reg = 0.02
    else:
        u_reg = 0.0
        v_reg = 0.0
        b_reg = 0.0
    tol = 0.001
    prev_rmse = np.inf
    obs_u, obs_i = np.where(rate_mat > 0)
    n_obs = len(obs_u)
    for _ in range(max_iter):
        order = rng.permutation(n_obs)
        for idx in order:
            u = obs_u[idx]
            i = obs_i[idx]
            u_old = U[u, :].copy()
            v_old = V[i, :].copy()
            bu_old = b_u[u, 0]
            bi_old = b_i[i, 0]
            pred = global_bias + bu_old + bi_old + np.dot(u_old, v_old)
            err = rate_mat[u, i] - pred
            U[u, :] += learning_rate * (err * v_old - u_reg * u_old)
            V[i, :] += learning_rate * (err * u_old - v_reg * v_old)
            b_u[u, 0] += learning_rate * (err - b_reg * bu_old)
            b_i[i, 0] += learning_rate * (err - b_reg * bi_old)
        pred_full = global_bias + b_u + b_i.T + U.dot(V.T)
        train_mask = rate_mat > 0
        train_rmse = np.sqrt(
            np.sum(((rate_mat - pred_full) * train_mask) ** 2) / float(np.sum(train_mask))
        )
        if abs(prev_rmse - train_rmse) < tol:
            break
        prev_rmse = train_rmse
    return (U, V, b_u, b_i, global_bias)


def demo():
    ratings = np.array([[5, 4, 0, 1], [4, 5, 1, 0], [1, 0, 5, 4], [0, 1, 4, 5]], dtype=float)
    U, V, bu, bi, b = my_recommender(ratings, 2, True)
    prediction = b + bu + bi.T + U @ V.T
    mask = ratings > 0
    return {
        "data": "synthetic ratings; zeros mean missing",
        "baseline_rmse": float(np.sqrt(np.mean((ratings[mask] - ratings[mask].mean()) ** 2))),
        "training_rmse": float(np.sqrt(np.mean((ratings[mask] - prediction[mask]) ** 2))),
        "predictions": prediction.tolist(),
    }


if __name__ == "__main__":
    import json

    print(json.dumps(demo(), indent=2))
