"""NumPy Gaussian mixture implementation from an individual CSE 6740 assignment."""
from typing import Tuple, Union, Callable
import numpy as np

def initialize_parameters(X: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    (m, n) = X.shape
    idx = np.random.choice(m, size=k, replace=False)
    MU = X[idx].copy()
    SIGMA = compute_sigma(X, MU)
    PI = np.full(k, 1.0 / k, dtype=float)
    return (MU, SIGMA, PI)

def compute_sigma(X: np.ndarray, MU: np.ndarray) -> np.ndarray:
    (m, n) = X.shape
    k = MU.shape[0]
    SIGMA = np.zeros((k, n, n), dtype=float)
    for j in range(k):
        diff = X - MU[j]
        SIGMA[j] = diff.T @ diff / m
    return SIGMA

def prob(x: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> Union[float, np.ndarray]:
    x = np.asarray(x, dtype=float)
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    single_input = x.ndim == 1
    X = x.reshape(1, -1) if single_input else x
    n = mu.shape[0]
    det = np.linalg.det(sigma)
    if det <= 0:
        sigma = sigma + np.eye(n) * 1e-06
        det = np.linalg.det(sigma)
    inv = np.linalg.inv(sigma)
    diff = X - mu
    quad = np.einsum('ij,jk,ik->i', diff, inv, diff)
    norm_const = 1.0 / ((2.0 * np.pi) ** (n / 2.0) * np.sqrt(det))
    p = norm_const * np.exp(-0.5 * quad)
    return float(p[0]) if single_input else p

def E_step(X: np.ndarray, MU: np.ndarray, SIGMA: np.ndarray, PI: np.ndarray, k: int) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    m = X.shape[0]
    PI_vec = np.asarray(PI, dtype=float).reshape(-1)
    if PI_vec.shape[0] != k:
        raise ValueError('PI has incompatible shape.')
    comp = np.zeros((k, m), dtype=float)
    for j in range(k):
        comp[j, :] = prob(X, MU[j], SIGMA[j])
    numer = PI_vec[:, None] * comp
    denom = np.sum(numer, axis=0, keepdims=True)
    denom = np.where(denom == 0, 1e-16, denom)
    r = numer / denom
    return r

def M_step(X: np.ndarray, r: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=float)
    r = np.asarray(r, dtype=float)
    (m, n) = X.shape
    if r.shape != (k, m):
        raise ValueError('Responsibilities r must have shape (k, m).')
    Nk = np.sum(r, axis=1, keepdims=True)
    Nk_safe = np.where(Nk == 0, 1e-16, Nk)
    new_MU = r @ X / Nk_safe
    new_PI = (Nk / m).reshape(-1)
    new_SIGMA = np.zeros((k, n, n), dtype=float)
    for j in range(k):
        diff = X - new_MU[j]
        w = r[j, :].reshape(m, 1)
        new_SIGMA[j] = diff.T @ (w * diff) / Nk_safe[j, 0]
    return (new_MU, new_SIGMA, new_PI)

def likelihood(X: np.ndarray, MU: np.ndarray, SIGMA: np.ndarray, PI: np.ndarray, k: int) -> float:
    X = np.asarray(X, dtype=float)
    m = X.shape[0]
    PI_vec = np.asarray(PI, dtype=float).reshape(-1)
    if PI_vec.shape[0] != k:
        raise ValueError('PI has incompatible shape.')
    probs = np.zeros((k, m), dtype=float)
    for j in range(k):
        probs[j, :] = prob(X, MU[j], SIGMA[j])
    mix = np.sum(PI_vec[:, None] * probs, axis=0)
    mix = np.where(mix == 0, 1e-16, mix)
    return float(np.sum(np.log(mix)))

def default_convergence(prev_likelihood, new_likelihood, conv_ctr, conv_ctr_cap=10):
    increase_convergence_ctr = abs(prev_likelihood) * 0.9 < abs(new_likelihood) < abs(prev_likelihood) * 1.1
    if increase_convergence_ctr:
        conv_ctr += 1
    else:
        conv_ctr = 0
    return (conv_ctr, conv_ctr > conv_ctr_cap)

def train_model(X: np.ndarray, k: int, convergence_function: Callable=default_convergence, initial_values: Tuple[np.ndarray, np.ndarray, np.ndarray]=None, max_iter: int=300) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or not X.size or not np.isfinite(X).all():
        raise ValueError("X must be a nonempty finite 2D array")
    if not isinstance(k, int) or not 1 <= k <= len(X):
        raise ValueError("k must be between 1 and the number of observations")
    if not isinstance(max_iter, int) or max_iter < 1:
        raise ValueError("max_iter must be a positive integer")
    if initial_values is None:
        (MU, SIGMA, PI) = initialize_parameters(X, k)
    else:
        (MU, SIGMA, PI) = initial_values
    prev_lkl = likelihood(X, MU, SIGMA, PI, k)
    conv_ctr = 0
    for _ in range(max_iter):
        r = E_step(X, MU, SIGMA, PI, k)
        (MU, SIGMA, PI) = M_step(X, r, k)
        new_lkl = likelihood(X, MU, SIGMA, PI, k)
        (conv_ctr, converged) = convergence_function(prev_lkl, new_lkl, conv_ctr)
        if not np.isfinite(new_lkl):
            raise FloatingPointError("Non-finite likelihood; check covariance and data scale")
        if converged:
            break
        prev_lkl = new_lkl
    else:
        raise RuntimeError("EM did not converge within max_iter")
    r = E_step(X, MU, SIGMA, PI, k)
    return (MU, SIGMA, PI, r)
