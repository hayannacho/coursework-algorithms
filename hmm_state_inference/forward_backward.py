import numpy as np


def smoothed_posteriors(q, Y):
    if not 0 < q < 1:
        raise ValueError("q must lie strictly between 0 and 1")
    Y = np.asarray(Y).flatten()
    T = len(Y)
    if not T:
        raise ValueError("At least one observation is required")
    A = np.array([[0.8, 0.2], [0.2, 0.8]], dtype=float)
    pi = np.array([0.2, 0.8], dtype=float)

    def emission(y):
        if y == 1:
            return np.array([q, 1 - q], dtype=float)
        elif y == -1:
            return np.array([1 - q, q], dtype=float)
        else:
            raise ValueError("Each observation must be either +1 or -1.")

    alpha = np.zeros((T, 2), dtype=float)
    c = np.zeros(T, dtype=float)
    alpha[0] = pi * emission(Y[0])
    c[0] = alpha[0].sum()
    alpha[0] /= c[0]
    for t in range(1, T):
        alpha[t] = alpha[t - 1] @ A * emission(Y[t])
        c[t] = alpha[t].sum()
        alpha[t] /= c[t]
    beta = np.zeros((T, 2), dtype=float)
    beta[T - 1] = np.array([1.0, 1.0], dtype=float)
    for t in range(T - 2, -1, -1):
        beta[t] = A @ (emission(Y[t + 1]) * beta[t + 1])
        beta[t] /= c[t + 1]
    gamma = alpha * beta
    gamma /= gamma.sum(axis=1, keepdims=True)
    return gamma


if __name__ == "__main__":
    import argparse, json

    p = argparse.ArgumentParser(description="Scaled forward-backward state inference")
    p.add_argument("observations", type=int, nargs="+")
    p.add_argument("--q", type=float, default=0.7)
    a = p.parse_args()
    try:
        print(json.dumps({"posterior": smoothed_posteriors(a.q, a.observations).tolist()}, indent=2))
    except ValueError as e:
        p.error(str(e))
