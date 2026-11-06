"""Paired statistics used for every reported comparison (manuscript Section 3.8)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import binomtest

BOOT_RESAMPLES = 20_000
BOOT_SEED = 20260611       # fixed, so every interval is reproducible and independent of call order
TIE_TOL = 1e-9             # |paired change| below this many percent counts as a tie


def paired_pct(policy: pd.Series, base: pd.Series) -> pd.Series:
    """100 (M_policy,s / M_base,s - 1) per seed; the two series are aligned on seed."""
    a, b = policy.align(base, join="inner")
    if len(a) != len(policy) or len(a) != len(base):
        raise ValueError("paired comparison needs the same seeds on both sides")
    return 100.0 * (a / b - 1.0)


def bootstrap_mean_ci(values, n: int = BOOT_RESAMPLES, seed: int = BOOT_SEED, level: float = 0.95):
    """Percentile bootstrap interval for the mean of paired values (resampling seeds with replacement)."""
    x = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n, x.size))
    means = x[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * (1 - level) / 2, 100 * (1 + level) / 2])
    return float(lo), float(hi)


def summarize(d: pd.Series) -> dict:
    lo, hi = bootstrap_mean_ci(d.values)
    up, down = int((d > TIE_TOL).sum()), int((d < -TIE_TOL).sum())
    return {"mean": float(d.mean()), "median": float(d.median()), "ci_lo": lo, "ci_hi": hi, "n": int(d.size),
            "n_increase": up, "n_decrease": down, "n_tie": int((d.abs() <= TIE_TOL).sum()),
            "sign_p": sign_test(up, down)}


def sign_test(n_up: int, n_down: int) -> float:
    """Exact two-sided sign test (binomial, p = 0.5) on the seeds that are not ties."""
    n = n_up + n_down
    return 1.0 if n == 0 else float(binomtest(n_up, n, 0.5, alternative="two-sided").pvalue)
