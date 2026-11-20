"""Fit two Gaussian components to reproducible, entirely synthetic 2D observations."""
import argparse
import json
from pathlib import Path
import numpy as np
from gmm import compute_sigma, likelihood, train_model


def make_data(seed=42):
    rng = np.random.default_rng(seed)
    return np.concatenate([
        rng.multivariate_normal([-3, 0], [[0.5, 0.1], [0.1, 0.7]], 300),
        rng.multivariate_normal([3, 1], [[0.7, -0.1], [-0.1, 0.4]], 300),
    ])


def run(seed=42):
    X = make_data(seed)
    means = X[[np.argmin(X[:, 0]), np.argmax(X[:, 0])]].copy()
    initial = (means, compute_sigma(X, means), np.full(2, 0.5))
    history = [likelihood(X, *initial, 2)]

    def convergence(previous, current, counter):
        history.append(current)
        return counter + 1, abs(current - previous) <= 1e-8 * (1 + abs(previous))

    mu, sigma, pi, responsibilities = train_model(X, 2, convergence, initial, max_iter=300)
    order = np.argsort(mu[:, 0])
    mu, sigma, pi, responsibilities = mu[order], sigma[order], pi[order], responsibilities[order]
    labels = responsibilities.argmax(axis=0)
    summary = {
        'data': 'synthetic two-component Gaussian mixture; no real observations',
        'seed': seed, 'observations': len(X), 'dimensions': X.shape[1],
        'components': 2, 'iterations': len(history) - 1,
        'initial_log_likelihood': history[0], 'final_log_likelihood': history[-1],
        'means': mu.tolist(), 'weights': pi.tolist(),
        'assigned_counts': np.bincount(labels, minlength=2).tolist(),
        'minimum_likelihood_increment': float(np.min(np.diff(history))),
        'max_responsibility_sum_error': float(np.max(np.abs(responsibilities.sum(axis=0) - 1))),
    }
    return summary, X, labels, history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--outdir', type=Path, default=Path('output'))
    args = parser.parse_args()
    try:
        summary, X, labels, history = run(args.seed)
    except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as error:
        parser.error(str(error))
    args.outdir.mkdir(parents=True, exist_ok=True)
    (args.outdir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    np.savetxt(args.outdir / 'assignments.csv', np.column_stack([X, labels]), delimiter=',',
               header='x,y,component', comments='', fmt=['%.10f', '%.10f', '%d'])
    np.savetxt(args.outdir / 'likelihood.csv', np.column_stack([np.arange(len(history)), history]),
               delimiter=',', header='iteration,log_likelihood', comments='', fmt=['%d', '%.10f'])
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
