"""Observation-grid sensitivity check (manuscript Section 3.9).

For the first six seeds of the main benchmark (3201-3206) and of the stiff-shaft
benchmark (6201-6206), under RR and WS, the true shaft stress is reconstructed on a
0.5 ms grid between the 2 ms control updates. Decision timing and control inputs are
unchanged: inside each 2 ms interval the recorded state is propagated in four exact
0.5 ms ZOH sub-steps under the same held input u_j and disturbance d_j. F5 from the
refined history is compared with F5 from the 2 ms grid.

Outputs: results/stress_sampling_check.csv and results/stress_sampling_check.md
Usage:   python check_stress_sampling.py [--out results]
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from servo_bench import config as C
from servo_bench.plant import shear_stress, zoh
from servo_bench.rainflow import rainflow, severity
from servo_bench.simulate import simulate

SUBSTEPS = 4


def refined_twist(tr: dict, axis: int) -> tuple[np.ndarray, float]:
    """Twist q_m - q_l on the 0.5 ms grid and the max mismatch between sub-stepped and recorded states."""
    mdl = tr["model"]
    Ad_s, BE_s = zoh(mdl.A_true[axis], np.column_stack([mdl.B_true[axis], mdl.E_true[axis]]), C.DT / SUBSTEPS)
    X = tr["x"][:-1, axis, :]                       # recorded states at the 2 ms grid, j = 0..N-1
    u = tr["u"][:, axis][:, None]
    d = tr["dist"][:, axis][:, None]
    cols = []
    for _ in range(SUBSTEPS):
        cols.append(X[:, 0] - X[:, 2])
        X = X @ Ad_s.T + u * BE_s[:, 0] + d * BE_s[:, 1]
    mismatch = float(np.max(np.abs(X - tr["x"][1:, axis, :])))   # 4 sub-steps must land on x_{j+1}
    return np.stack(cols, axis=1).ravel(), mismatch


def check(cfg) -> dict:
    tr = simulate(cfg, return_traces=True)["traces"]
    ks = tr["model"].ks_true
    f_coarse = f_fine = 0.0
    worst = 0.0
    for i in range(C.N_AXES):
        coarse = shear_stress(ks[i] * (tr["x"][:-1, i, 0] - tr["x"][:-1, i, 2]), cfg.shaft_diameter)
        tw, mm = refined_twist(tr, i)
        fine = shear_stress(ks[i] * tw, cfg.shaft_diameter)
        f_coarse += severity(*rainflow(coarse), cfg.tau_ref, 5)
        f_fine += severity(*rainflow(fine), cfg.tau_ref, 5)
        worst = max(worst, mm)
    return {"experiment": cfg.experiment, "policy": cfg.policy, "seed": cfg.seed, "F5_2ms": f_coarse,
            "F5_0p5ms": f_fine, "deviation_pct": 100.0 * (f_fine / f_coarse - 1.0),
            "substep_state_mismatch": worst}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    cfgs = []
    for pol in ("RR", "WS"):
        cfgs += [C.RunConfig("main", "baseline", pol, s) for s in C.SEEDS_MAIN[:6]]
        cfgs += [C.RunConfig("stiff", "ds4mm", pol, s, shaft_diameter=0.004, tau_ref=5e6) for s in C.SEEDS_STIFF[:6]]
    df = pd.DataFrame([check(c) for c in cfgs])
    os.makedirs(a.out, exist_ok=True)
    df.to_csv(os.path.join(a.out, "stress_sampling_check.csv"), index=False, float_format="%.10g")
    lines = ["### Observation-grid sensitivity: F5 on the 2 ms grid vs the 0.5 ms reconstruction", "",
             "| Configuration | Runs | Max abs deviation in F5 | Mean abs deviation | Max sub-step state mismatch |",
             "|---|---|---|---|---|"]
    for exp, label in (("main", "Baseline (d_s = 2 mm), seeds 3201-3206, RR and WS"),
                       ("stiff", "Stiff shaft (d_s = 4 mm), seeds 6201-6206, RR and WS")):
        sub = df[df.experiment == exp]
        lines.append(f"| {label} | {len(sub)} | {sub.deviation_pct.abs().max():.3f}% | "
                     f"{sub.deviation_pct.abs().mean():.3f}% | {sub.substep_state_mismatch.max():.1e} |")
        print(f"{label}: max |dF5| = {sub.deviation_pct.abs().max():.3f}%")
    with open(os.path.join(a.out, "stress_sampling_check.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
