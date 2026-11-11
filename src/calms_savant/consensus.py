"""Two explicit heuristics. Neither certifies absence of poisoned priors."""
from dataclasses import dataclass
import numpy as np
from .numerics import unit, nonnegative


@dataclass(frozen=True)
class ConsensusResult:
    directions: dict
    accepted: tuple
    rejected: tuple
    distances: np.ndarray
    scores: np.ndarray
    threshold: float
    method: str


def seed_consensus(candidates, *, method="medoid_mad", alpha=2.0,
                   mad_multiplier=3.5, minimum_radius=0.02):
    """Node IDs must already be curated semantic correspondences."""
    if len(candidates) < 3:
        raise ValueError("at least three candidate seeds required")
    keys = sorted(candidates[0])
    if not keys or any(set(c) != set(keys) for c in candidates):
        raise ValueError("all seeds require the same nonempty semantic node IDs")
    values = np.array([[unit(c[k]) for k in keys] for c in candidates])
    if values.ndim != 3:
        raise ValueError("all node directions must be vectors with the same dimension")
    count = len(values)
    # Euclidean singleton OT per corresponding node, averaged over nodes.
    dist = np.zeros((count, count))
    for i in range(count):
        for j in range(i):
            dist[i, j] = dist[j, i] = np.mean(np.linalg.norm(values[i]-values[j], axis=1)/2)
    medians = np.array([np.median(np.delete(dist[i], i)) for i in range(count)])
    if method == "paper_mean_std":
        alpha = nonnegative(alpha, "alpha")
        scores = medians
        threshold = float(scores.mean() + alpha*scores.std(ddof=0))
    elif method == "medoid_mad":
        multiplier = nonnegative(mad_multiplier, "mad_multiplier")
        radius = nonnegative(minimum_radius, "minimum_radius")
        medoid = int(np.argmin(dist.sum(axis=1)))
        scores = dist[medoid]
        center = float(np.median(scores))
        mad = float(np.median(abs(scores-center)))
        threshold = center + max(radius, multiplier*1.4826*mad)
    else:
        raise ValueError("unknown consensus method")
    accepted = tuple(int(i) for i in np.flatnonzero(scores <= threshold))
    rejected = tuple(int(i) for i in range(count) if i not in accepted)
    if len(accepted) <= count//2:
        raise ValueError("no strict majority cluster; abstaining")
    weights = 1 / np.maximum(medians[list(accepted)], 1e-9)
    directions = {k: unit(np.sum(values[list(accepted), index] * weights[:, None], axis=0))
                  for index, k in enumerate(keys)}
    return ConsensusResult(directions, accepted, rejected, dist, scores, threshold, method)
