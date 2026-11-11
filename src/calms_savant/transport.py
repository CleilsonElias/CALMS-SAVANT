"""Log-domain entropic OT, explicitly distinct from exact Wasserstein-1."""
from dataclasses import dataclass
import numpy as np
from .numerics import matrix, unit, logsumexp, nonnegative, positive_int


@dataclass(frozen=True)
class TransportResult:
    transport_cost: float
    regularized_objective: float
    marginal_error: float
    iterations: int
    converged: bool
    plan: np.ndarray


def ground_cost(x, y, metric="euclidean"):
    x, y = unit(matrix(x)), unit(matrix(y))
    if x.shape[1] != y.shape[1]:
        raise ValueError("embedding dimensions differ")
    dot = np.clip(x @ y.T, -1, 1)
    if metric == "euclidean":
        # Euclidean distance / 2 on the unit sphere: a metric in [0,1].
        return np.sqrt(np.maximum(0.0, 2 - 2 * dot)) / 2
    if metric == "cosine":
        # Not a metric: no triangle inequality claim.
        return (1 - dot) / 2
    raise ValueError("metric must be euclidean or cosine")


def sinkhorn(x, y, *, epsilon=0.05, iterations=1000, tolerance=1e-8,
             metric="euclidean"):
    epsilon = nonnegative(epsilon, "epsilon")
    tolerance = nonnegative(tolerance, "tolerance")
    iterations = positive_int(iterations, "iterations")
    if epsilon == 0 or tolerance == 0:
        raise ValueError("epsilon and tolerance must be positive")
    c = ground_cost(x, y, metric)
    n, m = c.shape
    if n == 1 or m == 1:
        p = np.full((n, m), 1.0 / (n * m))
        cost = float(np.sum(p * c))
        return TransportResult(cost, cost, 0.0, 0, True, p)
    la, lb = np.full(n, -np.log(n)), np.full(m, -np.log(m))
    kernel = -c / epsilon
    lu, lv = np.zeros(n), np.zeros(m)
    residual = float("inf")
    for iteration in range(1, iterations + 1):
        lu = la - logsumexp(kernel + lv[None, :], axis=1)
        lv = lb - logsumexp(kernel + lu[:, None], axis=0)
        logp = lu[:, None] + kernel + lv[None, :]
        p = np.exp(logp)
        residual = max(float(np.max(abs(p.sum(1) - 1/n))),
                       float(np.max(abs(p.sum(0) - 1/m))))
        if residual <= tolerance:
            break
    cost = float(np.sum(p * c))
    # KL(T || a tensor b), with 0 log 0 = 0. All marginals are uniform.
    kl = float(np.sum(p * (logp - la[:, None] - lb[None, :])))
    return TransportResult(cost, cost + epsilon * kl, residual, iteration,
                           residual <= tolerance, p)


def distance(x, y, *, mode="divergence", **kwargs):
    if mode not in ("divergence", "transport"):
        raise ValueError("mode must be divergence or transport")
    xy = sinkhorn(x, y, **kwargs)
    results = [xy]
    value = xy.transport_cost
    if mode == "divergence":
        xx, yy = sinkhorn(x, x, **kwargs), sinkhorn(y, y, **kwargs)
        results += [xx, yy]
        value = xy.regularized_objective - (xx.regularized_objective + yy.regularized_objective)/2
    if not all(r.converged for r in results):
        raise ArithmeticError("Sinkhorn did not converge; increase iterations or regularization")
    # A material negative value is a solver/assumption failure, not zero drift.
    if value < -1e-6 or not np.isfinite(value):
        raise ArithmeticError(f"invalid transport/divergence {value}")
    return max(0.0, float(value))


def anchor_drift(before, after, **kwargs):
    """Compare each prompt to itself. Pooled permutations cannot hide swaps."""
    if not before or set(before) != set(after):
        raise ValueError("nonempty matching anchor IDs required")
    values = {key: distance(before[key], after[key], **kwargs) for key in sorted(before)}
    return float(np.mean(list(values.values()))), values
