"""Run the 24 AP-calibration runs and the 912 evaluation runs (manuscript Table 3).

Outputs (under --out, default ./data):
  calibration/calibration_runs.csv   one row per calibration run (WS, seeds 2201-2224)
  calibration/ap_shares.json         AP target shares derived from the calibration runs
  runs.csv                           one row per evaluation run (912 rows)
  packets/<experiment>/<run_id>.csv.gz   per-packet ledger of every run (936 files)
  traces/main_baseline_<policy>_s3201.csv.gz   example time series used for a figure
  run_log.json                       software versions, worker count and wall time

Usage:  python run_experiments.py [--workers 2] [--out data]
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import platform
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

from servo_bench import __version__
from servo_bench import config as C
from servo_bench.simulate import STATUS_NAMES, simulate

GZ = {"method": "gzip", "mtime": 0}     # deterministic gzip bytes
CONFIG_FIELDS = ["experiment", "condition", "policy", "seed", "rho", "t_run", "slot_ms", "delay_factor",
                 "mismatch", "shaft_diameter", "tau_ref", "estimator"]


def packet_frame(packets: dict) -> pd.DataFrame:
    df = pd.DataFrame(packets)
    df["status"] = df["status"].map(STATUS_NAMES)
    return df


def run_one(args) -> dict:
    cfg, out = args
    res = simulate(cfg)
    rel = os.path.join("packets", cfg.experiment, f"{cfg.run_id}.csv.gz")
    path = os.path.join(out, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    packet_frame(res["packets"]).to_csv(path, index=False, float_format="%.9g", compression=GZ)
    row = {f: getattr(cfg, f) for f in CONFIG_FIELDS}
    row["run_id"] = cfg.run_id
    row.update(res["summary"])
    row["packets_file"] = rel.replace(os.sep, "/")
    return row


def run_batch(cfgs, out, workers):
    jobs = [(c, out) for c in cfgs]
    if workers <= 1:
        return [run_one(j) for j in jobs]
    with Pool(workers) as pool:
        rows = []
        for i, row in enumerate(pool.imap(run_one, jobs, chunksize=4)):
            rows.append(row)
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(jobs)} runs", flush=True)
        return rows


def ap_allocation(shares, n_slots: int) -> list[int]:
    p = np.asarray(shares, float)
    n = np.zeros(3, dtype=int)
    for k in range(n_slots):
        n[int(np.argmax((k + 1) * p - n))] += 1
    return n.tolist()


def save_traces(out: str) -> None:
    os.makedirs(os.path.join(out, "traces"), exist_ok=True)
    for pol in ("RR", "WS"):
        cfg = C.RunConfig("main", "baseline", pol, C.SEEDS_MAIN[0])
        tr = simulate(cfg, return_traces=True)["traces"]
        n = cfg.n_steps
        df = pd.DataFrame({"t_s": np.arange(n) * C.DT})
        for i in range(C.N_AXES):
            df[f"ref_rad_axis{i}"] = tr["ref"][:, i]
            df[f"q_load_rad_axis{i}"] = tr["x"][:n, i, 2]
            df[f"tau_MPa_axis{i}"] = tr["tau"][:, i] / 1e6
            df[f"u_Nm_axis{i}"] = tr["u"][:, i]
        df.to_csv(os.path.join(out, "traces", f"main_baseline_{pol}_s{cfg.seed}.csv.gz"),
                  index=False, float_format="%.9g", compression=GZ)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default="data")
    a = ap.parse_args()
    out = a.out
    os.makedirs(os.path.join(out, "calibration"), exist_ok=True)
    t0 = time.time()

    print("AP calibration: 24 WS runs on seeds 2201-2224", flush=True)
    cal = pd.DataFrame(run_batch(C.calibration_configs(), out, a.workers))
    cal.to_csv(os.path.join(out, "calibration", "calibration_runs.csv"), index=False, float_format="%.12g")
    share = np.stack([cal[f"attempted_axis{i}"] / cal["attempted"] for i in range(C.N_AXES)], axis=1)
    raw = share.mean(axis=0)
    shares = tuple(float(round(v, 5)) for v in raw)
    n_slots = C.RunConfig("main", "baseline", "AP", 0).n_slots
    info = {"seeds": list(C.SEEDS_CALIBRATION), "policy": "WS",
            "mean_request_share_raw": raw.tolist(), "ap_shares_rounded_5dp": list(shares),
            "ap_allocation_per_run": ap_allocation(shares, n_slots), "n_slots": n_slots}
    with open(os.path.join(out, "calibration", "ap_shares.json"), "w") as f:
        json.dump(info, f, indent=2)
    print(f"  AP shares {shares} -> allocation {info['ap_allocation_per_run']}", flush=True)

    cfgs = C.evaluation_configs(shares)
    assert len(cfgs) == 912, len(cfgs)
    print(f"Evaluation: {len(cfgs)} runs on {a.workers} worker(s)", flush=True)
    rows = run_batch(cfgs, out, a.workers)
    pd.DataFrame(rows).to_csv(os.path.join(out, "runs.csv"), index=False, float_format="%.12g")
    elapsed_runs = time.time() - t0

    save_traces(out)
    log = {"package_version": __version__, "python": sys.version.split()[0], "platform": platform.platform(),
           "numpy": np.__version__, "pandas": pd.__version__,
           "scipy": __import__("scipy").__version__, "workers": a.workers,
           "evaluation_runs": len(cfgs), "calibration_runs": len(cal),
           "wall_time_936_runs_s": round(elapsed_runs, 1)}
    with open(os.path.join(out, "run_log.json"), "w") as f:
        json.dump(log, f, indent=2)
    print(f"Done: 936 runs in {elapsed_runs:.1f} s", flush=True)


if __name__ == "__main__":
    main()
