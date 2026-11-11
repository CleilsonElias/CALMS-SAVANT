"""Strict finite inputs and unit directions; zero vectors are undefined."""
import numpy as np


def matrix(value, name="matrix"):
    out = np.asarray(value, dtype=np.float64)
    if out.ndim != 2 or min(out.shape) == 0 or not np.isfinite(out).all():
        raise ValueError(f"{name} must be a nonempty finite 2D array")
    return out


def unit(value):
    out = np.asarray(value, dtype=np.float64)
    if out.ndim not in (1, 2) or out.size == 0 or not np.isfinite(out).all():
        raise ValueError("directions must be finite vectors or a matrix")
    norm = np.linalg.norm(out, axis=-1, keepdims=True)
    if not np.isfinite(norm).all() or (norm < 1e-12).any():
        raise ValueError("zero or numerically degenerate direction")
    # Preserve already-normalized saved vectors, avoiding repeated roundoff drift.
    return np.where(np.abs(norm-1.0) < 1e-14, out, out/norm)


def nonnegative(value, name):
    v = float(value)
    if not np.isfinite(v) or v < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return v


def positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def logsumexp(x, axis):
    m = np.max(x, axis=axis, keepdims=True)
    return np.squeeze(m + np.log(np.exp(x - m).sum(axis=axis, keepdims=True)), axis=axis)
