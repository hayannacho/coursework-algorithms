import numpy as np


def compute_dist(X, centers):
    N, D = X.shape
    K = centers.shape[0]
    dist = np.sqrt(np.sum((X[:, np.newaxis, :] - centers[np.newaxis, :, :]) ** 2, axis=2))
    return dist


def mykmeans(digits, K, seed=42):
    digits = validate(digits, K)
    rng = np.random.RandomState(seed)
    N, D = digits.shape
    idx = rng.choice(N, K, replace=False)
    centers = digits[idx].astype(float)
    max_iter = 100
    for i in range(max_iter):
        old_centers = centers.copy()
        distances = compute_dist(digits, centers)
        assignments = np.argmin(distances, axis=1)
        for k in range(K):
            cluster_data = digits[assignments == k]
            if len(cluster_data) > 0:
                centers[k] = np.mean(cluster_data, axis=0)
            else:
                centers[k] = digits[rng.choice(N)]
        if np.all(old_centers == centers):
            break
    return (np.argmin(compute_dist(digits, centers), axis=1), centers)


def mykmedoids(digits, K, seed=42):
    digits = validate(digits, K)
    rng = np.random.RandomState(seed)
    N, D = digits.shape
    idx = rng.choice(N, K, replace=False)
    centers = digits[idx].astype(float)
    max_iter = 50
    for i in range(max_iter):
        old_centers = centers.copy()
        distances = compute_dist(digits, centers)
        assignments = np.argmin(distances, axis=1)
        for k in range(K):
            cluster_indices = np.where(assignments == k)[0]
            if len(cluster_indices) > 0:
                cluster_data = digits[cluster_indices]
                intra_dist = compute_dist(cluster_data, cluster_data)
                best_idx_in_cluster = np.argmin(np.sum(intra_dist, axis=1))
                centers[k] = cluster_data[best_idx_in_cluster]
            else:
                centers[k] = digits[rng.choice(N)]
        if np.all(old_centers == centers):
            break
    return (np.argmin(compute_dist(digits, centers), axis=1), centers)


def validate(data, k):
    data = np.asarray(data, dtype=float)
    if data.ndim != 2 or not data.size or not np.isfinite(data).all():
        raise ValueError("finite nonempty 2D data required")
    if type(k) is not int or not 1 <= k <= len(data):
        raise ValueError("k must be between 1 and number of points")
    return data


def demo():
    rng = np.random.RandomState(7)
    x = np.vstack([rng.normal([-3, 0], 0.2, (20, 2)), rng.normal([3, 0], 0.2, (20, 2))])
    datasets = {"clean": x, "with_outliers": np.vstack([x, [[0, 12], [0, -12]]])}
    result = {}
    for label, data in datasets.items():
        result[label] = {}
        for name, fn in [("kmeans", mykmeans), ("kmedoids", mykmedoids)]:
            a, c = fn(data, 2)
            result[label][name] = {
                "centers": c.tolist(),
                "squared_error": float(((data - c[a]) ** 2).sum()),
                "sum_distance": float(np.linalg.norm(data - c[a], axis=1).sum()),
            }
    return result


if __name__ == "__main__":
    import json

    print(json.dumps(demo(), indent=2))
