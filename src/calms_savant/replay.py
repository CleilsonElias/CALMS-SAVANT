"""Capped stratified sampling, independent of the evaluation attack budget."""
import math
import numpy as np
from .numerics import positive_int, nonnegative


def stratified_replay(trusted, rejected, uncertain, *, size, max_rejected_ratio=0.15, seed=0):
    size = positive_int(size, "size")
    ratio = nonnegative(max_rejected_ratio, "max_rejected_ratio")
    if ratio > 1:
        raise ValueError("ratio must be <= 1")
    rng = np.random.default_rng(seed)
    negative_count = min(size, math.floor(size*ratio)) if rejected else 0
    other = list(trusted) + list(uncertain)
    if not other and size > negative_count:
        raise ValueError("not enough non-rejected categories; refusing an all-poison replay")
    selected = [other[int(i)] for i in rng.integers(len(other), size=size-negative_count)] if other else []
    selected += [rejected[int(i)] for i in rng.integers(len(rejected), size=negative_count)] if negative_count else []
    rng.shuffle(selected)
    return selected
