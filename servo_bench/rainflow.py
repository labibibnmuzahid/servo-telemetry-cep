"""Three-point rainflow counting and the cycle severity index (manuscript Section 3.7, Appendix A).

The algorithm follows the simplified rainflow procedure of ASTM E1049 / Downing and
Socie (1982): consecutive duplicates are compressed, turning points are extracted
(end points kept), and points are pushed on a stack. With X the latest range and Y
the preceding range, a cycle of range Y is extracted whenever X >= Y; it counts as a
half cycle if it contains the first point still on the stack and as a full cycle
otherwise. Ranges left on the stack at the end count as half cycles.
"""
from __future__ import annotations

import numpy as np


def turning_points(series) -> np.ndarray:
    """Peaks and valleys of a series, with consecutive duplicates removed and end points kept."""
    x = np.asarray(series, dtype=float).ravel()
    if x.size == 0:
        return x
    keep = np.ones(x.size, dtype=bool)
    keep[1:] = x[1:] != x[:-1]
    x = x[keep]
    if x.size < 3:
        return x
    d = np.diff(x)
    interior = np.nonzero(d[:-1] * d[1:] < 0.0)[0] + 1
    idx = np.concatenate(([0], interior, [x.size - 1]))
    return x[idx]


def rainflow(series) -> tuple[np.ndarray, np.ndarray]:
    """Return (ranges, counts) with counts in {0.5, 1.0}."""
    pts = turning_points(series).tolist()
    stack: list[float] = []
    ranges: list[float] = []
    counts: list[float] = []
    for p in pts:
        stack.append(p)
        while len(stack) >= 3:
            x_rng = abs(stack[-1] - stack[-2])
            y_rng = abs(stack[-2] - stack[-3])
            if x_rng < y_rng:
                break
            if len(stack) == 3:
                ranges.append(y_rng)
                counts.append(0.5)
                stack.pop(0)
            else:
                ranges.append(y_rng)
                counts.append(1.0)
                del stack[-3:-1]
    for k in range(len(stack) - 1):
        ranges.append(abs(stack[k + 1] - stack[k]))
        counts.append(0.5)
    return np.asarray(ranges, float), np.asarray(counts, float)


def severity(ranges: np.ndarray, counts: np.ndarray, tau_ref: float, m: float) -> float:
    """F_m contribution sum n_c (a_c / tau_ref)^m with cycle amplitude a_c = range / 2 (Eq. 14)."""
    if ranges.size == 0:
        return 0.0
    return float(np.sum(counts * (0.5 * ranges / tau_ref) ** m))
