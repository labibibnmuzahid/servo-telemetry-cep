"""Automated verification suite: 114 numerical and physical assertions (manuscript Section 3.9).

Groups
  A (20) rainflow extraction: hand-countable benchmarks, invariances, independent re-implementation
  B (18) plant model: geometry, exact ZOH vs RK4, semigroup identities, Van Loan covariance
  C  (6) controller: published LQR gain and eigenvalues, Riccati residual, stability, saturation
  D (12) estimators: input replay, Kalman update vs batch Gaussian conditioning, covariance health
  E (14) exogenous processes, channel and scheduler logic
  F  (6) deterministic reproducibility (in-process, fresh process, archived rows)
  G (12) experiment design, packet conservation and per-axis summation over all 936 runs
  H  (2) algebraic equivalence of SA and RR
  I (24) re-run of the RR baseline for all 24 main seeds against the archived dataset

Requires the archive produced by run_experiments.py (default ./data).
Writes results/validation_report.txt; exits with status 1 if any check fails.
Usage:  python validate.py [--data data] [--out results]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from fractions import Fraction

import numpy as np
import pandas as pd
import scipy.linalg as sla

from servo_bench import config as C
from servo_bench.estimators import DelayedKF, replay
from servo_bench.exogenous import generate, streams
from servo_bench.plant import (continuous_matrices, kf_process_covariance, lqr_gain, natural_frequency,
                               nominal_params, reference_torque, van_loan, zoh)
from servo_bench.rainflow import rainflow, severity, turning_points
from servo_bench.schedulers import torque_variance_curve
from servo_bench.simulate import STATUS_NAMES, build_model, simulate

CHECKS: list = []
DATA = "data"
_cache: dict = {}


def check(code: str, desc: str):
    def deco(fn):
        CHECKS.append((code, desc, fn))
        return fn
    return deco


def runs() -> pd.DataFrame:
    if "runs" not in _cache:
        _cache["runs"] = pd.read_csv(os.path.join(DATA, "runs.csv"))
    return _cache["runs"]


def calib() -> pd.DataFrame:
    if "cal" not in _cache:
        _cache["cal"] = pd.read_csv(os.path.join(DATA, "calibration", "calibration_runs.csv"))
    return _cache["cal"]


def all_runs() -> pd.DataFrame:
    return pd.concat([runs(), calib()], ignore_index=True)


def packets(row) -> pd.DataFrame:
    return pd.read_csv(os.path.join(DATA, row["packets_file"]))


def row_of(exp, cond, pol, seed):
    df = runs()
    return df[(df.experiment == exp) & (df.condition == cond) & (df.policy == pol) & (df.seed == seed)].iloc[0]


def as_multiset(ranges, counts, nd=9):
    d = defaultdict(float)
    for r, c in zip(ranges, counts):
        d[round(float(r), nd)] += float(c)
    return dict(d)


NUMERIC_SKIP = {"seed", "packets_file", "run_id", "experiment", "condition", "policy", "estimator"}


def matches_archive(summary: dict, row) -> bool:
    for k, v in summary.items():
        a = float(row[k])
        if not np.isclose(float(v), a, rtol=1e-9, atol=1e-12):
            raise AssertionError(f"{k}: rerun {v!r} vs archive {a!r}")
    return True


def rng_signals(n=200, seed=7):
    rng = np.random.default_rng(seed)
    return [np.cumsum(rng.standard_normal(rng.integers(3, 400))) for _ in range(n)]


def rk4_step(A, B, E, x, u, d, dt, n=1000):
    h = dt / n
    f = lambda z: A @ z + B * u + E * d  # noqa: E731
    for _ in range(n):
        k1 = f(x); k2 = f(x + 0.5 * h * k1); k3 = f(x + 0.5 * h * k2); k4 = f(x + h * k3)
        x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return x


# ============================================================================ A: rainflow (20)
ASTM = [-2, 1, -3, 5, -1, 3, -4, 4, -2]


@check("A01", "ASTM E1049 worked example gives ranges 3/4/6/8/9 with counts 0.5/1.5/0.5/1.0/0.5")
def _():
    return as_multiset(*rainflow(ASTM)) == {3.0: 0.5, 4.0: 1.5, 6.0: 0.5, 8.0: 1.0, 9.0: 0.5}


@check("A02", "turning points of the ASTM example are the sequence itself")
def _():
    return turning_points(ASTM).tolist() == [float(v) for v in ASTM]


@check("A03", "consecutive duplicates are compressed before turning-point extraction")
def _():
    return turning_points([0, 1, 1, 1, 0]).tolist() == [0.0, 1.0, 0.0]


@check("A04", "plateaus at a reversal are reduced to one turning point")
def _():
    return turning_points([0, 2, 2, 1, 1, 3]).tolist() == [0.0, 2.0, 1.0, 3.0]


@check("A05", "a monotone ramp is one half cycle of the full range")
def _():
    r, c = rainflow(np.arange(11.0))
    return r.tolist() == [10.0] and c.tolist() == [0.5]


@check("A06", "a constant signal has no cycles and zero severity")
def _():
    r, c = rainflow(np.ones(50))
    return r.size == 0 and severity(r, c, 1.0, 5) == 0.0


@check("A07", "a two-point history is one half cycle")
def _():
    r, c = rainflow([0.0, 5.0])
    return r.tolist() == [5.0] and c.tolist() == [0.5]


@check("A08", "a nested inner loop is extracted as a full cycle: [0,4,1,3,0] -> {2: 1.0, 4: 1.0}")
def _():
    return as_multiset(*rainflow([0, 4, 1, 3, 0])) == {2.0: 1.0, 4.0: 1.0}


@check("A09", "a 5-period sine gives 4.5 equivalent cycles of range 2 and two half cycles of range 1")
def _():
    t = np.arange(400 * 5 + 1)
    r, c = rainflow(np.sin(2 * np.pi * t / 400))
    return abs(c[np.abs(r - 2) < 1e-9].sum() - 4.5) < 1e-12 and abs(c[np.abs(r - 1) < 1e-9].sum() - 1.0) < 1e-12


@check("A10", "every reversal is counted once: 2 * sum(counts) = turning points - 1 (200 random histories)")
def _():
    return all(abs(2 * rainflow(x)[1].sum() - (turning_points(x).size - 1)) < 1e-12 for x in rng_signals())


@check("A11", "no extracted range exceeds max - min of the history (200 random histories)")
def _():
    return all(rainflow(x)[0].max() <= x.max() - x.min() + 1e-12 for x in rng_signals())


@check("A12", "the overall range max - min is always counted (200 random histories)")
def _():
    return all(abs(rainflow(x)[0].max() - (x.max() - x.min())) < 1e-9 for x in rng_signals())


@check("A13", "scale invariance: ranges scale with the history, counts unchanged")
def _():
    ok = True
    for x in rng_signals(50):
        r1, c1 = rainflow(x)
        r2, c2 = rainflow(3.7 * x)
        ok &= np.allclose(r2, 3.7 * r1, rtol=1e-12) and np.array_equal(c1, c2)
    return ok


@check("A14", "severity obeys the power law F_m(c x) = c^m F_m(x)")
def _():
    ok = True
    for x in rng_signals(50):
        f1 = severity(*rainflow(x), 1.0, 5)
        f2 = severity(*rainflow(2.5 * x), 1.0, 5)
        ok &= np.isclose(f2, 2.5 ** 5 * f1, rtol=1e-10)
    return ok


@check("A15", "offset invariance: adding a constant does not change ranges or counts")
def _():
    ok = True
    for x in rng_signals(50):
        r1, c1 = rainflow(x)
        r2, c2 = rainflow(x + 17.0)
        ok &= np.allclose(r1, r2, rtol=0, atol=1e-9) and np.array_equal(c1, c2)
    return ok


@check("A16", "sign symmetry: the mirrored history has the same cycle multiset")
def _():
    return all(as_multiset(*rainflow(x)) == as_multiset(*rainflow(-x)) for x in rng_signals(50))


@check("A17", "amplitude convention: one full cycle of range 2 tau_ref gives F_m = 1 for m = 3, 5, 7")
def _():
    return all(np.isclose(severity(np.array([2 * 40e6]), np.array([1.0]), 40e6, m), 1.0) for m in (3, 5, 7))


def astm_reference(series):
    """Independent transcription of ASTM E1049-85 section 5.4.4, written with explicit
    point indices, a discarded-point list and an explicit starting point S."""
    vals = []
    for v in np.asarray(series, float):
        if vals and v == vals[-1]:
            continue
        vals.append(float(v))
    tp = []
    for k, v in enumerate(vals):
        if k == 0 or k == len(vals) - 1 or (vals[k] - vals[k - 1]) * (vals[k + 1] - vals[k]) < 0:
            tp.append(v)
    alive: list[int] = []              # indices of points read and not yet discarded
    S = 0                              # index of the starting point
    out = []
    for idx in range(len(tp)):         # step 1: read next peak or valley
        alive.append(idx)
        while len(alive) >= 3:         # step 2: X and Y from the three most recent undiscarded points
            i3, i2, i1 = alive[-3], alive[-2], alive[-1]
            X = abs(tp[i1] - tp[i2])
            Y = abs(tp[i2] - tp[i3])
            if X < Y:                  # step 3(a)
                break
            if S in (i3, i2):          # step 4 -> step 5: Y contains S
                out.append((Y, 0.5))
                alive.remove(i3)       # discard the first point of Y
                S = i2                 # move S to the second point of Y
            else:                      # step 4: count Y as one cycle, discard its peak and valley
                out.append((Y, 1.0))
                alive.remove(i3)
                alive.remove(i2)
    for a, b in zip(alive[:-1], alive[1:]):   # step 6
        out.append((abs(tp[b] - tp[a]), 0.5))
    return out


@check("A18", "package rainflow equals an independent ASTM E1049 transcription (200 random histories)")
def _():
    for x in rng_signals(seed=11):
        a = sorted(zip(*[v.tolist() for v in rainflow(x)]))
        b = sorted(astm_reference(x))
        if len(a) != len(b) or not np.allclose(np.array(a), np.array(b), rtol=0, atol=1e-12):
            return False
    return True


@check("A19", "cycle counts are always 0.5 or 1.0")
def _():
    return all(set(np.unique(rainflow(x)[1]).tolist()) <= {0.5, 1.0} for x in rng_signals(50))


@check("A20", "empty and single-sample histories produce no cycles")
def _():
    return rainflow([])[0].size == 0 and rainflow([3.0])[0].size == 0


# ============================================================================ B: plant (18)
JM, JL, KS = nominal_params()


@check("B01", "nominal motor inertia J_m = 0.00324 kg m^2 from r = 0.09 m, m_m = 0.8 kg")
def _():
    return np.isclose(JM, 0.00324, rtol=1e-12)


@check("B02", "nominal load inertia J_l = 0.00567 kg m^2 from m_l = 1.4 kg")
def _():
    return np.isclose(JL, 0.00567, rtol=1e-12)


@check("B03", "shaft stiffness k_s = G pi d^4 / (32 L) = 0.62046455 N m/rad")
def _():
    return round(KS, 8) == 0.62046455


@check("B04", "torsional natural frequency f_f = 2.76092 Hz")
def _():
    return round(natural_frequency(JM, JL, KS), 5) == 2.76092


@check("B05", "reference torque T_ref = tau_ref pi d^3 / 16 = 0.0628319 N m")
def _():
    return round(reference_torque(40e6, 0.002), 7) == 0.0628319


@check("B06", "stiff shaft (d_s = 4 mm): k_s = 9.93 N m/rad and f_f = 11.04 Hz")
def _():
    jm, jl, ks = nominal_params(0.004)
    return round(ks, 2) == 9.93 and round(natural_frequency(jm, jl, ks), 2) == 11.04


@check("B07", "stiff configuration keeps T_ref: T_ref(5 MPa, 4 mm) = T_ref(40 MPa, 2 mm)")
def _():
    return np.isclose(reference_torque(5e6, 0.004), reference_torque(40e6, 0.002), rtol=1e-12)


def zoh_vs_rk4(jm, jl, ks, seed):
    A, B, E = continuous_matrices(jm, jl, ks)
    Ad, BE = zoh(A, np.column_stack([B, E]), C.DT)
    rng = np.random.default_rng(seed)
    x, u, d = rng.standard_normal(4) * [0.5, 2, 0.5, 2], rng.uniform(-0.3, 0.3), rng.uniform(-0.03, 0.03)
    exact = Ad @ x + BE[:, 0] * u + BE[:, 1] * d
    ref = rk4_step(A, B, E, x, u, d, C.DT)
    return np.max(np.abs(exact - ref)) <= 1e-10 * max(1.0, np.max(np.abs(ref)))


@check("B08", "exact ZOH step equals an independent RK4 solution (nominal plant)")
def _():
    return zoh_vs_rk4(JM, JL, KS, 1)


for _i in range(3):
    def _mk(i=_i):
        def fn():
            f = generate(C.RunConfig("main", "baseline", "RR", 3201)).factors[i]
            return zoh_vs_rk4(JM * f[0], JL * f[1], KS * f[2], 10 + i)
        return fn
    check(f"B{9 + _i:02d}", f"exact ZOH step equals RK4 for perturbed axis {_i} of seed 3201")(_mk())


@check("B12", "matrix-exponential semigroup: exp(A 2dt) = exp(A dt)^2")
def _():
    A, _, _ = continuous_matrices(JM, JL, KS)
    return np.allclose(sla.expm(2 * C.DT * A), sla.expm(C.DT * A) @ sla.expm(C.DT * A), rtol=1e-13, atol=1e-15)


@check("B13", "input integral semigroup: B_d(2dt) = A_d B_d + B_d")
def _():
    A, B, _ = continuous_matrices(JM, JL, KS)
    Ad, Bd = zoh(A, B[:, None], C.DT)
    _, Bd2 = zoh(A, B[:, None], 2 * C.DT)
    return np.allclose(Bd2, Ad @ Bd + Bd, rtol=1e-12, atol=1e-16)


@check("B14", "four 0.5 ms sub-steps compose exactly to one 2 ms step (state and input)")
def _():
    A, B, _ = continuous_matrices(JM, JL, KS)
    Ad, Bd = zoh(A, B[:, None], C.DT)
    As, Bs = zoh(A, B[:, None], C.DT / 4)
    acc = sum(np.linalg.matrix_power(As, k) @ Bs for k in range(4))
    return np.allclose(np.linalg.matrix_power(As, 4), Ad, rtol=1e-13, atol=1e-15) and np.allclose(acc, Bd, rtol=1e-12)


@check("B15", "open-loop torsional eigenvalue pair lies within 0.5% of 2 pi f_f")
def _():
    A, _, _ = continuous_matrices(JM, JL, KS)
    w = np.max(np.abs(np.linalg.eigvals(A).imag))
    return abs(w / (2 * np.pi * natural_frequency(JM, JL, KS)) - 1) < 0.005


@check("B16", "sign conventions: motor torque accelerates the motor, d_L decelerates the load")
def _():
    A, B, E = continuous_matrices(JM, JL, KS)
    _, BE = zoh(A, np.column_stack([B, E]), C.DT)
    return BE[1, 0] > 0 and BE[3, 1] < 0


@check("B17", "Van Loan Q_d equals direct Simpson quadrature of exp(As) Q_c exp(A^T s)")
def _():
    A, B, E = continuous_matrices(JM, JL, KS)
    Qc = 0.4 * C.PULSE_AMP[0] ** 2 * np.outer(E, E) + 1e-5 * np.outer(B, B)
    s = np.linspace(0, C.DT, 2001)
    vals = np.array([sla.expm(A * t) @ Qc @ sla.expm(A * t).T for t in s])
    w = np.ones(s.size); w[1:-1:2] = 4; w[2:-1:2] = 2
    quad = (s[1] - s[0]) / 3 * np.tensordot(w, vals, axes=1)
    Qd = van_loan(A, Qc, C.DT)
    return np.allclose(Qd, quad, rtol=1e-8, atol=1e-8 * np.abs(Qd).max()) and np.allclose(Qd, Qd.T)


@check("B18", "discrete process covariances are symmetric positive definite (all axes, both shafts)")
def _():
    ok = True
    for d in (0.002, 0.004):
        jm, jl, ks = nominal_params(d)
        for i in range(3):
            Q = kf_process_covariance(i, jm, jl, ks)
            np.linalg.cholesky(Q)
            ok &= np.allclose(Q, Q.T) and np.linalg.eigvalsh(Q).min() > 0
    return ok


# ============================================================================ C: controller (6)
A_N, B_N, _ = continuous_matrices(JM, JL, KS)
K_N = lqr_gain(A_N, B_N)


@check("C01", "LQR gain equals the published K = [3.312653, 0.158591, 1.883500, 0.462091]")
def _():
    return np.allclose(np.round(K_N, 6), [3.312653, 0.158591, 1.883500, 0.462091], atol=0)


@check("C02", "discrete closed-loop eigenvalues equal 0.979376 +/- 0.037782i and 0.967839 +/- 0.021856i")
def _():
    Ad, Bd = zoh(A_N, B_N[:, None], C.DT)
    ev = np.linalg.eigvals(Ad - Bd @ K_N[None, :])
    got = sorted((round(e.real, 6), round(abs(e.imag), 6)) for e in ev)
    return got == sorted([(0.979376, 0.037782)] * 2 + [(0.967839, 0.021856)] * 2)


@check("C03", "continuous Riccati residual is below 1e-9 relative to Q")
def _():
    Bm = B_N[:, None]
    P = sla.solve_continuous_are(A_N, Bm, np.diag(C.Q_LQR), np.array([[1.0]]))
    res = A_N.T @ P + P @ A_N - P @ Bm @ Bm.T @ P + np.diag(C.Q_LQR)
    return np.abs(res).max() < 1e-9 * max(C.Q_LQR)


@check("C04", "closed loop is stable for every perturbed axis of every seed (12% and 25% mismatch)")
def _():
    worst = 0.0
    for cfg in [C.RunConfig("main", "baseline", "RR", s) for s in C.SEEDS_MAIN] + \
               [C.RunConfig("operating", "mismatch25", "RR", s, mismatch=0.25) for s in C.SEEDS_OPERATING]:
        mdl = build_model(cfg, generate(cfg))
        for i in range(3):
            worst = max(worst, np.max(np.abs(np.linalg.eigvals(mdl.Ad_true[i] - np.outer(mdl.Bd_true[i], mdl.K)))))
    return worst < 1.0


@check("C05", "re-synthesised LQR stabilises the stiff shaft (nominal and all perturbed axes)")
def _():
    worst = 0.0
    for s in C.SEEDS_STIFF:
        cfg = C.RunConfig("stiff", "ds4mm", "RR", s, shaft_diameter=0.004, tau_ref=5e6)
        mdl = build_model(cfg, generate(cfg))
        for i in range(3):
            worst = max(worst, np.max(np.abs(np.linalg.eigvals(mdl.Ad_true[i] - np.outer(mdl.Bd_true[i], mdl.K)))))
    return worst < 1.0


@check("C06", "applied motor torque never exceeds the 0.30 N m saturation (RR, WS, SO traces)")
def _():
    return all(np.abs(simulate(C.RunConfig("main", "baseline", p, 3201), return_traces=True)["traces"]["u"]).max()
               <= C.U_MAX + 1e-15 for p in ("RR", "WS", "SO"))


# ============================================================================ D: estimators (12)
AD_N, BD_N = zoh(A_N, B_N[:, None], C.DT)
BD_N = BD_N[:, 0]
RV = np.diag(np.square(C.SIGMA))


@check("D01", "input replay equals the closed form A^n y + sum A^(n-1-l) B u_(s+l)")
def _():
    rng = np.random.default_rng(3)
    y, u = rng.standard_normal(4), rng.uniform(-0.3, 0.3, size=(60, 3))
    s, j = 12, 41
    closed = np.linalg.matrix_power(AD_N, j - s) @ y + sum(
        np.linalg.matrix_power(AD_N, j - s - 1 - l) @ BD_N * u[s + l, 1] for l in range(j - s))
    return np.allclose(replay(y, AD_N, BD_N, u[:, 1], s, j), closed, rtol=1e-12, atol=1e-14)


@check("D02", "input replay with zero inputs equals A_d^n y")
def _():
    y = np.array([0.1, -0.4, 0.05, 0.3])
    return np.allclose(replay(y, AD_N, BD_N, np.zeros(40), 3, 33), np.linalg.matrix_power(AD_N, 30) @ y, rtol=1e-13)


@check("D03", "with a matched model, no noise and no disturbance, replay recovers the true state exactly")
def _():
    rng = np.random.default_rng(5)
    u = rng.uniform(-0.3, 0.3, 80)
    x = [rng.standard_normal(4)]
    for l in range(79):
        x.append(AD_N @ x[-1] + BD_N * u[l])
    return np.allclose(replay(x[17], AD_N, BD_N, u, 17, 63), x[63], rtol=1e-12, atol=1e-14)


def kf_fresh(Rv=RV):
    jm, jl, ks = JM, JL, KS
    Qd = np.stack([kf_process_covariance(i, jm, jl, ks) for i in range(3)])
    return DelayedKF(AD_N, BD_N, Qd, Rv)


@check("D04", "Kalman measurement update equals batch Gaussian conditioning")
def _():
    kf = kf_fresh()
    rng = np.random.default_rng(8)
    L = rng.standard_normal((4, 4)) * 1e-3
    P0 = L @ L.T + 1e-8 * np.eye(4)
    m0 = rng.standard_normal(4) * 1e-2
    kf.m[0], kf.P[0] = m0, P0
    y = m0 + rng.standard_normal(4) * 1e-3
    kf.accept(0, 0, y, 0, np.zeros((1, 3)))
    G = P0 @ np.linalg.inv(P0 + RV)
    return np.allclose(kf.m[0], m0 + G @ (y - m0), rtol=1e-9, atol=1e-15) and \
        np.allclose(kf.P[0], P0 - G @ P0, rtol=1e-7, atol=1e-18)


@check("D05", "two sequential updates equal one batch update with stacked measurements")
def _():
    kf = kf_fresh()
    rng = np.random.default_rng(9)
    L = rng.standard_normal((4, 4)) * 1e-3
    P0 = L @ L.T + 1e-8 * np.eye(4)
    m0 = np.zeros(4)
    kf.m[0], kf.P[0] = m0, P0
    y1, y2 = rng.standard_normal(4) * 1e-3, rng.standard_normal(4) * 1e-3
    kf.accept(0, 0, y1, 0, np.zeros((1, 3)))
    kf.accept(0, 0, y2, 0, np.zeros((1, 3)))
    H = np.vstack([np.eye(4), np.eye(4)])
    R = sla.block_diag(RV, RV)
    S = H @ P0 @ H.T + R
    G = P0 @ H.T @ np.linalg.inv(S)
    m_b = m0 + G @ (np.concatenate([y1, y2]) - H @ m0)
    P_b = P0 - G @ H @ P0
    return np.allclose(kf.m[0], m_b, rtol=1e-8, atol=1e-15) and np.allclose(kf.P[0], P_b, rtol=1e-6, atol=1e-18)


@check("D06", "Joseph-form covariance stays symmetric positive definite over 500 delayed updates")
def _():
    kf = kf_fresh()
    rng = np.random.default_rng(10)
    u = rng.uniform(-0.3, 0.3, size=(20000, 3))
    gen = 0
    for _ in range(500):
        gen += int(rng.integers(1, 25))
        kf.accept(1, gen, rng.standard_normal(4) * 1e-2, gen, u)
        P = kf.P[1]
        if not (np.allclose(P, P.T, rtol=0, atol=1e-18) and np.linalg.eigvalsh(P).min() > 0):
            return False
    return True


@check("D07", "a measurement update never increases uncertainty: P_prior - P_post is positive semidefinite")
def _():
    kf = kf_fresh()
    rng = np.random.default_rng(12)
    u = rng.uniform(-0.3, 0.3, size=(500, 3))
    ok = True
    for gen in range(5, 400, 13):
        m, P = kf.m[2].copy(), kf.P[2].copy()
        for l in range(int(kf.g[2]), gen):
            P = AD_N @ P @ AD_N.T + kf.Qd[2]
        kf.accept(2, gen, rng.standard_normal(4) * 1e-2, gen, u)
        ok &= np.linalg.eigvalsh(P - kf.P[2]).min() > -1e-12 * np.abs(P).max()
    return ok


@check("D08", "Kalman limits: R -> 0 returns the measurement, R -> infinity returns the prior")
def _():
    y = np.array([0.01, -0.02, 0.03, 0.04])
    kf0 = kf_fresh(Rv=1e-24 * np.eye(4))
    kf0.P[0] = 1e-6 * np.eye(4)
    kf0.accept(0, 0, y, 0, np.zeros((1, 3)))
    kfi = kf_fresh(Rv=1e12 * np.eye(4))
    kfi.P[0] = 1e-6 * np.eye(4)
    kfi.accept(0, 0, y, 0, np.zeros((1, 3)))
    return np.allclose(kf0.m[0], y, rtol=1e-9) and np.allclose(kfi.m[0], 0.0, atol=1e-15)


@check("D09", "delayed KF = propagate to the sample time, update, then replay inputs to the arrival step")
def _():
    kf = kf_fresh()
    rng = np.random.default_rng(13)
    u = rng.uniform(-0.3, 0.3, size=(30, 3))
    y = rng.standard_normal(4) * 1e-2
    out = kf.accept(0, 6, y, 11, u)
    m, P = np.zeros(4), RV.copy()
    for l in range(6):
        m = AD_N @ m + BD_N * u[l, 0]
        P = AD_N @ P @ AD_N.T + kf.Qd[0]
    G = P @ np.linalg.inv(P + RV)
    m = m + G @ (y - m)
    for l in range(6, 11):
        m = AD_N @ m + BD_N * u[l, 0]
    return np.allclose(out, m, rtol=1e-9, atol=1e-14)


@check("D10", "TU variance curve equals c^T A_d^n R_v (A_d^n)^T c computed with matrix powers")
def _():
    V = torque_variance_curve(AD_N, RV, KS, 60)
    c = KS * np.array([1.0, 0, -1.0, 0])
    ref = [c @ np.linalg.matrix_power(AD_N, n) @ RV @ np.linalg.matrix_power(AD_N, n).T @ c for n in range(61)]
    return np.allclose(V, ref, rtol=1e-10)


@check("D11", "TU variance curve rises monotonically over the first 40 steps (80 ms)")
def _():
    V = torque_variance_curve(AD_N, RV, KS, 40)
    return bool(np.all(np.diff(V) > 0))


@check("D12", "between accepted packets the estimate follows the nominal model under applied inputs")
def _():
    out = simulate(C.RunConfig("main", "baseline", "RR", 3201), return_traces=True)
    tr, pk = out["traces"], out["packets"]
    acc_steps = set(pk["arr_step"][pk["status"] == 1].tolist())
    xh, u = tr["x_hat"], tr["u"]
    worst = 0.0
    for j in range(len(u) - 1):
        if j + 1 in acc_steps:
            continue
        pred = xh[j] @ AD_N.T + u[j][:, None] * BD_N[None, :]
        worst = max(worst, np.max(np.abs(pred - xh[j + 1])))
    return worst < 1e-12


# ============================================================================ E: exogenous, channel, schedulers (14)
def exo_equal(a, b):
    return (np.array_equal(a.factors, b.factors) and np.array_equal(a.phases, b.phases)
            and np.array_equal(a.ref, b.ref) and np.array_equal(a.dist, b.dist)
            and np.array_equal(a.meas_noise, b.meas_noise) and np.array_equal(a.lost, b.lost)
            and np.array_equal(a.delay, b.delay))


@check("E01", "exogenous generation is deterministic for a given seed")
def _():
    cfg = C.RunConfig("main", "baseline", "WS", 3207)
    return exo_equal(generate(cfg), generate(cfg))


@check("E02", "exogenous draws do not depend on the scheduling policy (RR, WS, SO, AP, TU)")
def _():
    base = generate(C.RunConfig("main", "baseline", "RR", 3210))
    return all(exo_equal(base, generate(C.RunConfig("main", "baseline", p, 3210))) for p in ("WS", "SO", "AP", "TU"))


@check("E03", "delays lie in {2..8} steps outside congestion and {15..30} steps inside it")
def _():
    ok = True
    for s in C.SEEDS_MAIN:
        ex = generate(C.RunConfig("main", "baseline", "RR", s))
        d, cg = ex.delay, ex.congested
        ok &= d[~cg].min() >= 2 and d[~cg].max() <= 8 and d[cg].min() >= 15 and d[cg].max() <= 30
    return ok


@check("E04", "congestion flags equal 2.2 < (t mod 3.2) < 2.7 s in exact arithmetic (6, 10, 20 ms slots)")
def _():
    ok = True
    for slot_ms in (6.0, 10.0, 20.0):
        cfg = C.RunConfig("operating", "x", "RR", 4201, slot_ms=slot_ms)
        ex = generate(cfg)
        for k in range(cfg.n_slots):
            t = Fraction(k * cfg.slot_steps, 500)
            ph = t - Fraction(16, 5) * (t // Fraction(16, 5))
            ok &= bool(ex.congested[k]) == (Fraction(11, 5) < ph < Fraction(27, 10))
    return ok


@check("E05", "observed packet-loss count is within 4 sigma of the 2% / 15% channel model (24 seeds)")
def _():
    obs = exp = var = 0.0
    for s in C.SEEDS_MAIN:
        ex = generate(C.RunConfig("main", "baseline", "RR", s))
        p = np.where(ex.congested, C.LOSS_CONG, C.LOSS_BASE)
        obs += ex.lost.sum(); exp += p.sum(); var += (p * (1 - p)).sum()
    return abs(obs - exp) <= 4 * np.sqrt(var)


@check("E06", "doubled-delay condition doubles every delay and keeps the loss pattern")
def _():
    a = generate(C.RunConfig("operating", "x", "RR", 4201))
    b = generate(C.RunConfig("operating", "x", "RR", 4201, delay_factor=2))
    return np.array_equal(b.delay, 2 * a.delay) and np.array_equal(a.lost, b.lost)


@check("E07", "mismatch levels share one set of unit draws: deviations scale linearly and stay in bounds")
def _():
    f0 = generate(C.RunConfig("operating", "x", "RR", 4202, mismatch=0.0)).factors
    f12 = generate(C.RunConfig("operating", "x", "RR", 4202, mismatch=0.12)).factors
    f25 = generate(C.RunConfig("operating", "x", "RR", 4202, mismatch=0.25)).factors
    return (np.all(f0 == 1.0) and np.allclose((f25 - 1) / 0.25, (f12 - 1) / 0.12, rtol=1e-12)
            and np.abs(f25 - 1).max() <= 0.25 and np.abs(f12 - 1).max() <= 0.12)


@check("E08", "references start at zero and never exceed their amplitudes a = (0.60, 0.35, 0.22) rad")
def _():
    ex = generate(C.RunConfig("main", "baseline", "RR", 3201))
    return np.all(ex.ref[0] == 0) and np.all(np.abs(ex.ref).max(axis=0) <= np.asarray(C.REF_AMP))


@check("E09", "load pulses: onsets within +/-0.15 s of 1 + 0.37 i + 2.4 n, D in [0.25, 0.55] s, amplitude +/-A_i")
def _():
    ok = True
    for s in C.SEEDS_MAIN[:6] + C.SEEDS_LONG[:2]:
        t_run = 64.0 if s in C.SEEDS_LONG else 16.0
        ex = generate(C.RunConfig("x", "x", "RR", s, t_run=t_run))
        last_end = defaultdict(lambda: -1.0)
        for i, t0, d, a in sorted(ex.pulses, key=lambda p: (p[0], p[1])):
            n = round((t0 - 1 - 0.37 * i) / 2.4)
            ok &= abs(t0 - (1 + 0.37 * i + 2.4 * n)) <= 0.15 + 1e-12
            ok &= 0.25 <= d <= 0.55 and np.isclose(abs(a), C.PULSE_AMP[i]) and t0 >= last_end[i]
            last_end[i] = t0 + d
    return ok


@check("E10", "disturbance = pulses + 0.002 z with z_j = 0.97 z_(j-1) + 0.03 w_j (explicit recursion)")
def _():
    cfg = C.RunConfig("main", "baseline", "RR", 3203)
    ex = generate(cfg)
    w = streams(cfg.seed)[3].standard_normal(size=(cfg.n_steps, 3))
    z = np.zeros_like(w)
    for j in range(cfg.n_steps):
        z[j] = (0.97 * z[j - 1] if j else 0.0) + 0.03 * w[j]
    t = np.arange(cfg.n_steps) * C.DT
    pulse = np.zeros_like(w)
    for i, t0, d, a in ex.pulses:
        msk = (t >= t0) & (t < t0 + d)
        pulse[msk, i] += a * np.sin(np.pi * (t[msk] - t0) / d) ** 2
    return np.max(np.abs(ex.dist - pulse - 0.002 * z)) < 1e-15


@check("E11", "RR requests axes in the cyclic order 0 -> 1 -> 2 in all 24 main runs")
def _():
    df = runs()
    return all((packets(r).axis.values == np.arange(1600) % 3).all()
               for _, r in df[(df.experiment == "main") & (df.policy == "RR")].iterrows())


@check("E12", "6h service guard: between requests no axis waits over 7 slots; all are served by slot 7 (SO-B, WS-B)")
def _():
    # Bound: the guard fires at a gap of 6 slots; S values of different axes are distinct after
    # start-up, so at most one slot of tie-break can follow. At t = 0 all axes share the initial
    # request time -dt, so the first guarded service of the last axis can fall on slot 7.
    df = runs()
    for _, r in df[(df.experiment == "main") & df.policy.isin(["SO-B", "WS-B"])].iterrows():
        ax = packets(r).axis.values
        for i in range(3):
            idx = np.nonzero(ax == i)[0]
            if idx[0] > 7 or (idx.size > 1 and np.diff(idx).max() > 7):
                return False
    return True


@check("E13", "AP shares equal the rounded calibration mean and every AP run allocates exactly that split")
def _():
    with open(os.path.join(DATA, "calibration", "ap_shares.json")) as f:
        info = json.load(f)
    cal = calib()
    raw = [float((cal[f"attempted_axis{i}"] / cal.attempted).mean()) for i in range(3)]
    df = runs()
    ap = df[(df.experiment == "main") & (df.policy == "AP")]
    return ([round(v, 5) for v in raw] == info["ap_shares_rounded_5dp"]
            and all(ap[f"attempted_axis{i}"].eq(info["ap_allocation_per_run"][i]).all() for i in range(3)))


@check("E14", "stale rule: accepted samples are strictly newer per axis; stale ones are not (all RA main runs)")
def _():
    df = runs()
    for _, r in df[(df.experiment == "main") & (df.policy == "RA")].iterrows():
        pk = packets(r)
        pk = pk[pk.status.isin(["accepted", "stale"])].sort_values(["arr_step", "gen_step"])
        last = {0: -1, 1: -1, 2: -1}
        for ax, gen, st in zip(pk.axis, pk.gen_step, pk.status):
            if st == "accepted":
                if gen <= last[ax]:
                    return False
                last[ax] = gen
            elif gen > last[ax]:
                return False
    return True


# ============================================================================ F: reproducibility (6)
def same_result(a, b):
    return a["summary"] == b["summary"] and all(
        np.array_equal(a["packets"][k], b["packets"][k], equal_nan=True) for k in a["packets"])


@check("F01", "repeated in-process runs are bit-identical (WS, seed 3201)")
def _():
    cfg = C.RunConfig("main", "baseline", "WS", 3201)
    return same_result(simulate(cfg), simulate(cfg))


@check("F02", "repeated in-process runs are bit-identical with the delayed Kalman filter")
def _():
    cfg = C.RunConfig("kalman", "delayed_kf", "WS", 3201, estimator="kf")
    return same_result(simulate(cfg), simulate(cfg))


@check("F03", "a fresh Python process reproduces the in-process summary exactly")
def _():
    code = ("import json; from servo_bench.config import RunConfig; from servo_bench.simulate import simulate; "
            "print(json.dumps(simulate(RunConfig('main','baseline','WS',3205))['summary']))")
    here = os.path.dirname(os.path.abspath(__file__))
    out = subprocess.run([sys.executable, "-c", code], cwd=here, capture_output=True, text=True, check=True)
    return json.loads(out.stdout) == simulate(C.RunConfig("main", "baseline", "WS", 3205))["summary"]


@check("F04", "re-run of the 64 s horizon (WS, seed 5201) matches the archived row")
def _():
    return matches_archive(simulate(C.RunConfig("long_horizon", "t64s", "WS", 5201, t_run=64.0))["summary"],
                           row_of("long_horizon", "t64s", "WS", 5201))


@check("F05", "re-run of the 6 ms slot condition (WS, seed 4201) matches the archived row")
def _():
    return matches_archive(simulate(C.RunConfig("operating", "slot6ms", "WS", 4201, slot_ms=6.0))["summary"],
                           row_of("operating", "slot6ms", "WS", 4201))


@check("F06", "re-run of the stiff shaft (RA, seed 6201) matches the archived row")
def _():
    cfg = C.RunConfig("stiff", "ds4mm", "RA", 6201, shaft_diameter=0.004, tau_ref=5e6)
    return matches_archive(simulate(cfg)["summary"], row_of("stiff", "ds4mm", "RA", 6201))


# ============================================================================ G: design, conservation, summation (12)
@check("G01", "run counts match the study design: 216 + 360 + 144 + 48 + 72 + 72 = 912, plus 24 calibration")
def _():
    n = runs().groupby("experiment").size().to_dict()
    return n == {"main": 216, "operating": 360, "rho_sweep": 144, "long_horizon": 48, "stiff": 72,
                 "kalman": 72} and len(runs()) == 912 and len(calib()) == 24


@check("G02", "every cell has its block's 24 seeds and a same-seed RR baseline")
def _():
    df = runs()
    block = {"main": C.SEEDS_MAIN, "operating": C.SEEDS_OPERATING, "rho_sweep": C.SEEDS_MAIN,
             "long_horizon": C.SEEDS_LONG, "stiff": C.SEEDS_STIFF, "kalman": C.SEEDS_MAIN}
    for (e, c, p), sub in df.groupby(["experiment", "condition", "policy"]):
        if sorted(sub.seed) != list(block[e]):
            return False
        be, bc = ("main", "baseline") if e == "rho_sweep" else (e, c)
        if sorted(df[(df.experiment == be) & (df.condition == bc) & (df.policy == "RR")].seed) != list(block[e]):
            return False
    return True


@check("G03", "every run attempted exactly one packet per slot")
def _():
    ok = True
    for _, r in all_runs().iterrows():
        cfg = C.RunConfig(r.experiment, r.condition, r.policy, int(r.seed), t_run=r.t_run, slot_ms=r.slot_ms)
        ok &= int(r.attempted) == cfg.n_slots
    return ok


@check("G04", "packet conservation N_attempted = accepted + stale + lost + pending in all 936 runs")
def _():
    d = all_runs()
    return bool((d.attempted == d.accepted + d.stale + d.lost + d.pending).all())


@check("G05", "packet conservation holds per axis in all 936 runs")
def _():
    d = all_runs()
    return all(((d[f"attempted_axis{i}"] == d[f"accepted_axis{i}"] + d[f"stale_axis{i}"] + d[f"lost_axis{i}"]
                 + d[f"pending_axis{i}"])).all() for i in range(3))


@check("G06", "per-axis request counts sum to the total in all 936 runs")
def _():
    d = all_runs()
    return bool((d.attempted == d.attempted_axis0 + d.attempted_axis1 + d.attempted_axis2).all())


for _m in (3, 5, 7):
    def _mk(m=_m):
        def fn():
            d = all_runs()
            return bool(np.allclose(d[f"F{m}"], d[f"F{m}_axis0"] + d[f"F{m}_axis1"] + d[f"F{m}_axis2"], rtol=1e-10))
        return fn
    check(f"G{4 + _m // 2 + 2:02d}", f"per-axis F{_m} values sum to the reported total in all 936 runs")(_mk())


@check("G10", "packet logs have one row per slot and status counts equal the run summaries (936 files)")
def _():
    for _, r in all_runs().iterrows():
        pk = packets(r)
        if len(pk) != int(r.attempted):
            return False
        vc = pk.status.value_counts()
        for name in ("accepted", "stale", "lost", "pending"):
            if int(vc.get(name, 0)) != int(r[name]):
                return False
        for i in range(3):
            if int((pk.axis == i).sum()) != int(r[f"attempted_axis{i}"]):
                return False
    return True


@check("G11", "packet timing: delivered = generation + delay, lost never arrive, pending arrive after the run")
def _():
    df = runs()
    for _, r in df[df.experiment.isin(["main", "operating"])].iterrows():
        pk = packets(r)
        n = int(round(r.t_run / C.DT))
        dl = pk[pk.status.isin(["accepted", "stale"])]
        if not ((dl.arr_step == dl.gen_step + dl.delay_steps) & (dl.arr_step < n)).all():
            return False
        lo = pk[pk.status == "lost"]
        pe = pk[pk.status == "pending"]
        if not ((lo.arr_step == -1).all() and (pe.arr_step >= n).all()):
            return False
    return True


@check("G12", "alignment flags exist only for accepted packets and their total matches the run summary")
def _():
    df = runs()
    for _, r in df[df.experiment == "main"].iterrows():
        pk = packets(r)
        acc = pk.status == "accepted"
        if not (pk.aligned[acc].isin([0, 1]).all() and (pk.aligned[~acc] == -1).all()):
            return False
        if int((pk.aligned == 1).sum()) != int(r.aligned_corrections):
            return False
    return True


# ============================================================================ H: SA = RR (2)
@check("H01", "SA and RR archived summaries are identical for all 24 main seeds")
def _():
    df = runs()
    cols = [c for c in df.columns if c not in NUMERIC_SKIP | {"policy"}]
    a = df[(df.experiment == "main") & (df.policy == "SA")].set_index("seed")[cols].sort_index()
    b = df[(df.experiment == "main") & (df.policy == "RR")].set_index("seed")[cols].sort_index()
    return a.equals(b)


@check("H02", "SA and RR packet ledgers are identical for all 24 main seeds")
def _():
    return all(packets(row_of("main", "baseline", "SA", s)).equals(packets(row_of("main", "baseline", "RR", s)))
               for s in C.SEEDS_MAIN)


# ============================================================================ I: archived baseline (24)
for _k, _s in enumerate(C.SEEDS_MAIN):
    def _mk(s=_s):
        def fn():
            return matches_archive(simulate(C.RunConfig("main", "baseline", "RR", s))["summary"],
                                   row_of("main", "baseline", "RR", s))
        return fn
    check(f"I{_k + 1:02d}", f"RR re-run for seed {_s} matches the archived dataset")(_mk())


# ============================================================================ runner
def main() -> None:
    global DATA
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    DATA = a.data
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    lines, n_pass = [], 0
    for code, desc, fn in CHECKS:
        try:
            ok = bool(fn())
            msg = ""
        except Exception as exc:  # noqa: BLE001 - every failure is reported, none is hidden
            ok, msg = False, f"  ({type(exc).__name__}: {exc})"
        n_pass += ok
        line = f"[{'PASS' if ok else 'FAIL'}] {code}  {desc}{msg}"
        lines.append(line)
        print(line, flush=True)
    total = len(CHECKS)
    footer = f"\n{n_pass} / {total} checks passed in {time.time() - t0:.1f} s"
    print(footer)
    with open(os.path.join(a.out, "validation_report.txt"), "w") as f:
        f.write("Verification suite for the stress-weighted telemetry benchmark\n\n" + "\n".join(lines) + footer + "\n")
    sys.exit(0 if n_pass == total else 1)


if __name__ == "__main__":
    main()
