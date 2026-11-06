"""Regenerate every table, statistic and figure from the archived runs.

Reads the archive written by run_experiments.py and writes
  results/tables/<name>.csv   raw numbers
  results/tables/<name>.md    formatted table (Markdown)
  results/tables/<name>.tex   formatted table (LaTeX, booktabs)
  results/key_numbers.json    headline statistics in machine-readable form
  results/summary.md          every table in one readable file
  results/figures/fig*.pdf / .png

All paired changes are 100 (M_policy,s / M_RR,s - 1) within seed s; intervals are
95 % percentile bootstrap intervals of the mean over 20,000 resamples of the paired
seeds, plus an exact two-sided sign test on the seeds that are not ties (servo_bench/stats.py).

Usage:  python analyze.py [--data data] [--out results]
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from servo_bench import config as C  # noqa: E402
from servo_bench.plant import natural_frequency, nominal_params  # noqa: E402
from servo_bench.stats import BOOT_RESAMPLES, BOOT_SEED, bootstrap_mean_ci, paired_pct, summarize  # noqa: E402

MAIN_ORDER = ["RR", "SA", "RA", "WS", "WS-B", "SO-B", "AP", "TU", "SO"]
CONDITION_LABELS = {
    ("main", "baseline"): "Main benchmark (10 ms slots, 12% mismatch)",
    ("operating", "slot6ms"): "Fast telemetry (6 ms slots)",
    ("operating", "slot20ms"): "Slow telemetry (20 ms slots)",
    ("operating", "delay2x"): "Doubled channel delay",
    ("operating", "mismatch25"): "25% parameter mismatch",
    ("operating", "mismatch0"): "Nominal plant (no mismatch)",
    ("long_horizon", "t64s"): "Extended 64 s horizon",
    ("kalman", "delayed_kf"): "Delayed Kalman filter",
    ("stiff", "ds4mm"): "Stiff transmission (d_s = 4 mm)",
}

# figure styling: reference palette, light surface (validated: first three categorical slots)
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
S1, S2 = "#2a78d6", "#eb6834"


# ----------------------------------------------------------------------------- helpers
def metric(df, exp, cond, pol, col="F5") -> pd.Series:
    sub = df[(df.experiment == exp) & (df.condition == cond) & (df.policy == pol)]
    if sub.empty:
        raise KeyError((exp, cond, pol))
    return sub.set_index("seed")[col].sort_index()


def paired(df, exp, cond, pol, col="F5", base="RR", base_exp=None, base_cond=None):
    d = paired_pct(metric(df, exp, cond, pol, col),
                   metric(df, base_exp or exp, base_cond or cond, base, col))
    return d, summarize(d)


def pct(x, nd=2):
    return f"{x:+.{nd}f}%"


def ci(s, nd=2):
    return f"[{s['ci_lo']:+.{nd}f}%, {s['ci_hi']:+.{nd}f}%]"


def fp(p):
    return f"{p:.2g}" if p >= 1e-3 else f"{p:.1e}"


def idt(s):
    return f"{s['n_increase']} / {s['n_decrease']} / {s['n_tie']}"


class Writer:
    def __init__(self, out):
        self.dir = os.path.join(out, "tables")
        os.makedirs(self.dir, exist_ok=True)
        self.sections: list[str] = []

    @staticmethod
    def _tex_escape(s: str) -> str:
        for a, b in (("\\", r"\textbackslash{}"), ("%", r"\%"), ("&", r"\&"), ("_", r"\_"), ("#", r"\#")):
            s = s.replace(a, b)
        return s

    def table(self, name: str, title: str, raw: pd.DataFrame, shown: pd.DataFrame, note: str = "") -> None:
        raw.to_csv(os.path.join(self.dir, f"{name}.csv"), index=False, float_format="%.10g")
        cols = [str(c) for c in shown.columns]
        md = [f"### {title}", "", "| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
        for _, r in shown.iterrows():
            md.append("| " + " | ".join(str(v) for v in r.values) + " |")
        if note:
            md += ["", note]
        text = "\n".join(md) + "\n"
        with open(os.path.join(self.dir, f"{name}.md"), "w") as f:
            f.write(text)
        self.sections.append(text)
        tex = [r"% " + title, r"\begin{tabular}{@{}" + "l" * len(cols) + r"@{}}", r"\toprule",
               " & ".join(self._tex_escape(c) for c in cols) + r" \\", r"\midrule"]
        for _, r in shown.iterrows():
            tex.append(" & ".join(self._tex_escape(str(v)) for v in r.values) + r" \\")
        tex += [r"\bottomrule", r"\end{tabular}"]
        with open(os.path.join(self.dir, f"{name}.tex"), "w") as f:
            f.write("\n".join(tex) + "\n")


# ----------------------------------------------------------------------------- tables
def table_main_outcomes(df, W, K):
    rows = []
    for p in MAIN_ORDER:
        sub = df[(df.experiment == "main") & (df.policy == p)]
        rows.append({"policy": p, "F5": sub.F5.mean(), "rmse_mrad": 1e3 * sub.rmse_rad.mean(),
                     "peak_tau_MPa": sub.peak_tau_MPa.mean(), "mean_age_ms": sub.mean_age_ms.mean(),
                     "p99_age_ms": sub.p99_age_ms.mean(), "accepted": sub.accepted.mean(),
                     "stale": sub.stale.mean(), "lost": sub.lost.mean(), "pending": sub.pending.mean(),
                     "attempted": sub.attempted.mean()})
    raw = pd.DataFrame(rows)
    shown = pd.DataFrame({
        "Policy": raw.policy,
        "F5": raw.F5.map(lambda v: f"{v:.4f}" if v < 100 else f"{v:.1f}"),
        "RMSE (mrad)": raw.rmse_mrad.map("{:.2f}".format),
        "Peak abs(tau) (MPa)": raw.peak_tau_MPa.map("{:.1f}".format),
        "Mean age (ms)": raw.mean_age_ms.map("{:.1f}".format),
        "P99 age (ms)": raw.p99_age_ms.map("{:.1f}".format),
        "Accepted": raw.accepted.map("{:.1f}".format),
        "Stale": raw.stale.map("{:.1f}".format),
        "Lost": raw.lost.map("{:.1f}".format),
    })
    W.table("t1_main_outcomes", f"Main benchmark outcomes averaged across {C.N_SEEDS} paired seeds "
            f"(seeds {C.SEEDS_MAIN[0]}-{C.SEEDS_MAIN[-1]})",
            raw, shown, "Every trial attempted exactly one packet per slot (1600 attempts per 16 s run).")
    K["main_means"] = raw.set_index("policy")[["F5", "rmse_mrad", "peak_tau_MPa", "mean_age_ms", "p99_age_ms",
                                               "accepted", "stale"]].to_dict(orient="index")
    K["attempted_per_run_main"] = sorted(set(df[df.experiment == "main"].attempted.astype(int)))


def table_paired_effects(df, W, K):
    rows, shown = [], []
    for p in ["WS", "WS-B", "RA", "SO-B", "AP", "TU", "SA", "SO"]:
        d, s = paired(df, "main", "baseline", p)
        _, r = paired(df, "main", "baseline", p, col="rmse_rad")
        rows.append({"policy": p, **{f"dF5_{k}": v for k, v in s.items()}, "dRMSE_pct_mean": r["mean"]})
        shown.append({"Policy": p, "dF5 (%)": pct(s["mean"]), "95% bootstrap interval": ci(s),
                      "Increase / decrease / tie": idt(s), "Sign-test p": fp(s["sign_p"]),
                      "dRMSE (%)": f"{r['mean']:+.4f}%"})
        K.setdefault("paired_vs_RR_main", {})[p] = {"dF5": s, "dRMSE_pct_mean": r["mean"]}
    W.table("t2_paired_effects", f"Paired percentage change in F5 relative to RR (main benchmark, {C.N_SEEDS} seeds)",
            pd.DataFrame(rows), pd.DataFrame(shown),
            "Increase / decrease / tie counts seeds where the policy's F5 is above / below / equal to RR's; "
            "the sign test is exact, two-sided, on the seeds that are not ties.")


def table_exponents(df, W, K):
    rows, shown = [], []
    for p in ["WS", "WS-B", "RA", "SO-B", "AP", "TU"]:
        rec, sh = {"policy": p}, {"Policy": p}
        for m in (3, 5, 7):
            _, s = paired(df, "main", "baseline", p, col=f"F{m}")
            rec.update({f"dF{m}_mean": s["mean"], f"dF{m}_lo": s["ci_lo"], f"dF{m}_hi": s["ci_hi"]})
            sh[f"dF{m}"] = f"{pct(s['mean'])} {ci(s)}"
            K.setdefault("exponents", {}).setdefault(p, {})[f"F{m}"] = s
        rows.append(rec)
        shown.append(sh)
    W.table("t3_cycle_exponents", "Paired change relative to RR for cycle exponents m = 3, 5, 7 (main benchmark)",
            pd.DataFrame(rows), pd.DataFrame(shown))


def table_per_axis(df, W, K):
    rr = df[(df.experiment == "main") & (df.policy == "RR")]
    share = {i: float(rr[f"F5_axis{i}"].sum() / rr.F5.sum()) for i in range(3)}
    K["rr_axis_share_of_F5"] = share
    rows, shown = [], []
    for p in ["WS", "RA", "SO-B", "AP", "SO"]:
        rec, sh = {"policy": p}, {"Policy": p}
        for i in range(3):
            _, s = paired(df, "main", "baseline", p, col=f"F5_axis{i}")
            rec.update({f"axis{i}_mean": s["mean"], f"axis{i}_lo": s["ci_lo"], f"axis{i}_hi": s["ci_hi"]})
            sh[f"Axis {i} dF5"] = f"{pct(s['mean'])} {ci(s)}"
            K.setdefault("per_axis", {}).setdefault(p, {})[f"axis{i}"] = s
        rows.append(rec)
        shown.append(sh)
    note = (f"Share of total RR severity (pooled over {C.N_SEEDS} seeds): "
            + ", ".join(f"axis {i} {100 * share[i]:.2f}%" for i in range(3)) + ".")
    W.table("t4_per_axis", "Per-axis paired change in F5 relative to RR (main benchmark)",
            pd.DataFrame(rows), pd.DataFrame(shown), note)


def table_starvation(df, W, K):
    rows, shown = [], []
    for p in ["RR", "RA", "WS", "SO", "SO-B"]:
        sub = df[(df.experiment == "main") & (df.policy == p)]
        rec = {"policy": p}
        for i in range(3):
            rec[f"requests_axis{i}"] = sub[f"attempted_axis{i}"].mean()
            rec[f"max_age_ms_axis{i}_mean"] = sub[f"max_age_ms_axis{i}"].mean()
            rec[f"max_age_ms_axis{i}_max"] = sub[f"max_age_ms_axis{i}"].max()
        rec["peak_tau_MPa_mean"] = sub.peak_tau_MPa.mean()
        rec["peak_tau_MPa_max"] = sub.peak_tau_MPa.max()
        rec["max_age_ms_any_axis_max"] = max(rec[f"max_age_ms_axis{i}_max"] for i in range(3))
        rows.append(rec)
        shown.append({"Policy": p,
                      "Requests (axis 0/1/2)": " / ".join(f"{rec[f'requests_axis{i}']:.1f}" for i in range(3)),
                      "Max receiver age, mean over seeds (ms)":
                          " / ".join(f"{rec[f'max_age_ms_axis{i}_mean']:.0f}" for i in range(3)),
                      "Max receiver age, worst seed (ms)":
                          " / ".join(f"{rec[f'max_age_ms_axis{i}_max']:.0f}" for i in range(3)),
                      "Peak abs(tau) mean / worst (MPa)": f"{rec['peak_tau_MPa_mean']:.1f} / {rec['peak_tau_MPa_max']:.1f}"})
        K.setdefault("starvation", {})[p] = rec
    W.table("t5_starvation", "Telemetry starvation and service guards (main benchmark)",
            pd.DataFrame(rows), pd.DataFrame(shown))


def table_ap(df, W, K, ap_info):
    d_ws_ap, s_ws_ap = paired(df, "main", "baseline", "WS", base="AP")
    _, s_ap = paired(df, "main", "baseline", "AP")
    ap_runs = df[(df.experiment == "main") & (df.policy == "AP")]
    alloc = [float(ap_runs[f"attempted_axis{i}"].mean()) for i in range(3)]
    raw = pd.DataFrame([{
        "ap_share_axis0": ap_info["ap_shares_rounded_5dp"][0], "ap_share_axis1": ap_info["ap_shares_rounded_5dp"][1],
        "ap_share_axis2": ap_info["ap_shares_rounded_5dp"][2], "ap_alloc_axis0": alloc[0],
        "ap_alloc_axis1": alloc[1], "ap_alloc_axis2": alloc[2], "dF5_AP_vs_RR": s_ap["mean"],
        "dF5_AP_vs_RR_lo": s_ap["ci_lo"], "dF5_AP_vs_RR_hi": s_ap["ci_hi"], "dF5_WS_vs_AP": s_ws_ap["mean"],
        "dF5_WS_vs_AP_lo": s_ws_ap["ci_lo"], "dF5_WS_vs_AP_hi": s_ws_ap["ci_hi"],
        "seeds_WS_above_AP": s_ws_ap["n_increase"]}])
    shown = pd.DataFrame([
        {"Quantity": f"AP target shares (axis 0/1/2), from {len(C.SEEDS_CALIBRATION)} WS calibration runs",
         "Value": " / ".join(f"{v:.5f}" for v in ap_info["ap_shares_rounded_5dp"])},
        {"Quantity": "AP packets per 16 s run (axis 0/1/2)", "Value": " / ".join(f"{v:.0f}" for v in alloc)},
        {"Quantity": "AP vs RR, dF5", "Value": f"{pct(s_ap['mean'])} {ci(s_ap)}"},
        {"Quantity": "WS vs AP, dF5", "Value": f"{pct(s_ws_ap['mean'])} {ci(s_ws_ap)}"},
        {"Quantity": "Seeds with WS above AP", "Value": f"{s_ws_ap['n_increase']} of {s_ws_ap['n']}"},
    ])
    W.table("t6_ap_vs_ws", "Bandwidth reallocation (AP) versus state-dependent timing (WS)", raw, shown)
    K["ap"] = {"shares": ap_info["ap_shares_rounded_5dp"], "allocation": alloc, "AP_vs_RR": s_ap, "WS_vs_AP": s_ws_ap}


def table_packet_diagnostics(df, W, K, data_dir):
    def load(pol):
        out = {}
        for seed in C.SEEDS_MAIN:
            f = df[(df.experiment == "main") & (df.policy == pol) & (df.seed == seed)].packets_file.iloc[0]
            pk = pd.read_csv(os.path.join(data_dir, f))
            out[seed] = pk[(pk.status == "accepted") & (pk.axis == 0)]
        return out
    rr, ws = load("RR"), load("WS")
    thr = float(np.percentile(np.abs(np.concatenate([rr[s].torque_true_Nm.values for s in C.SEEDS_MAIN])), 80))
    rec = {}
    for name, logs in (("RR", rr), ("WS", ws)):
        high = {s: logs[s][np.abs(logs[s].torque_true_Nm) > thr] for s in C.SEEDS_MAIN}
        rec[name] = {
            "high": pd.Series({s: len(h) for s, h in high.items()}),
            "aligned": pd.Series({s: int((h.aligned == 1).sum()) for s, h in high.items()}),
            "arrivals": pd.Series({s: len(logs[s]) for s, h in high.items()}),
            "frac": sum(int((h.aligned == 1).sum()) for h in high.values()) / sum(len(h) for h in high.values()),
            "est_err_mNm": 1e3 * float(np.mean(np.concatenate(
                [np.abs(logs[s].torque_est_before_Nm - logs[s].torque_true_Nm).values for s in C.SEEDS_MAIN]))),
        }
    diff = rec["WS"]["aligned"] - rec["RR"]["aligned"]
    lo, hi = bootstrap_mean_ci(diff.values)
    raw = pd.DataFrame([{
        "threshold_mNm": 1e3 * thr,
        "accepted_axis0_RR": rec["RR"]["arrivals"].mean(), "accepted_axis0_WS": rec["WS"]["arrivals"].mean(),
        "high_torsion_RR": rec["RR"]["high"].mean(), "high_torsion_WS": rec["WS"]["high"].mean(),
        "aligned_high_RR": rec["RR"]["aligned"].mean(), "aligned_high_WS": rec["WS"]["aligned"].mean(),
        "aligned_high_diff_mean": float(diff.mean()), "aligned_high_diff_lo": lo, "aligned_high_diff_hi": hi,
        "aligned_fraction_RR": rec["RR"]["frac"], "aligned_fraction_WS": rec["WS"]["frac"],
        "abs_torque_est_err_mNm_RR": rec["RR"]["est_err_mNm"], "abs_torque_est_err_mNm_WS": rec["WS"]["est_err_mNm"]}])
    r = raw.iloc[0]
    shown = pd.DataFrame([
        {"Quantity": "High-torsion threshold: 80th percentile of abs(T) at accepted axis-0 arrivals under RR",
         "RR": f"{r.threshold_mNm:.3f} mN m", "WS": ""},
        {"Quantity": "Accepted axis-0 packets per run", "RR": f"{r.accepted_axis0_RR:.2f}", "WS": f"{r.accepted_axis0_WS:.2f}"},
        {"Quantity": "High-torsion axis-0 arrivals per run", "RR": f"{r.high_torsion_RR:.2f}", "WS": f"{r.high_torsion_WS:.2f}"},
        {"Quantity": "... of which the command change reinforced the twist", "RR": f"{r.aligned_high_RR:.2f}",
         "WS": f"{r.aligned_high_WS:.2f}"},
        {"Quantity": "Paired difference WS - RR in reinforcing high-torsion arrivals", "RR": "",
         "WS": f"{r.aligned_high_diff_mean:+.2f} [{lo:+.2f}, {hi:+.2f}]"},
        {"Quantity": "Share of high-torsion arrivals that reinforced the twist", "RR": f"{100 * r.aligned_fraction_RR:.1f}%",
         "WS": f"{100 * r.aligned_fraction_WS:.1f}%"},
        {"Quantity": "Mean abs torque-estimate error just before an axis-0 update", "RR": f"{r.abs_torque_est_err_mNm_RR:.3f} mN m",
         "WS": f"{r.abs_torque_est_err_mNm_WS:.3f} mN m"},
    ])
    W.table("t7_packet_diagnostics", f"Packet diagnostics on axis 0 (main benchmark, {C.N_SEEDS} seeds)", raw, shown,
            "A correction reinforces the twist when the change in saturated motor command caused by the update "
            "has the same sign as the true shaft twist q_m - q_l at the arrival step.")
    K["packet_diagnostics"] = raw.iloc[0].to_dict()


def table_robustness(df, W, K):
    rows, shown = [], []
    for (exp, cond), label in CONDITION_LABELS.items():
        rec, sh = {"experiment": exp, "condition": cond, "label": label}, {"Operating condition": label}
        for p in ("WS", "RA", "TU"):
            try:
                _, s = paired(df, exp, cond, p)
            except KeyError:
                if p != "TU":
                    sh[f"{p} dF5 [95% CI]"] = "-"
                continue
            rec.update({f"{p}_mean": s["mean"], f"{p}_lo": s["ci_lo"], f"{p}_hi": s["ci_hi"],
                        f"{p}_n_increase": s["n_increase"], f"{p}_n_decrease": s["n_decrease"], f"{p}_n_tie": s["n_tie"]})
            if p != "TU":
                sh[f"{p} dF5 [95% CI]"] = f"{pct(s['mean'])} {ci(s)}"
            if p == "WS":
                sh["WS incr. / decr. / tie"] = idt(s)
                sh["WS sign-test p"] = fp(s["sign_p"])
                sh["WS median"] = pct(s["median"])
            rec[f"{p}_sign_p"] = s["sign_p"]
            rec[f"{p}_median"] = s["median"]
            K.setdefault("robustness", {}).setdefault(label, {})[p] = s
        sh["Seeds"] = str(len(metric(df, exp, cond, "RR")))
        rows.append(rec)
        shown.append(sh)
    W.table("t8_robustness", "Paired change in F5 relative to RR across operating conditions", pd.DataFrame(rows),
            pd.DataFrame(shown), "Each condition is compared with RR on the same seeds and in the same condition. "
            "Under the delayed Kalman filter, TU vs RR is reported in key_numbers.json.")


def table_rho(df, W, K):
    rows, shown = [], []
    for rho in sorted(list(C.RHO_SWEEP) + [C.RHO]):
        if rho == C.RHO:
            d, s = paired(df, "main", "baseline", "WS")
            share = df[(df.experiment == "main") & (df.policy == "WS")]
        else:
            d, s = paired(df, "rho_sweep", f"rho{rho:g}", "WS", base_exp="main", base_cond="baseline")
            share = df[(df.experiment == "rho_sweep") & (df.condition == f"rho{rho:g}")]
        sh0 = float((share.attempted_axis0 / share.attempted).mean())
        rows.append({"rho": rho, **{f"dF5_{k}": v for k, v in s.items()}, "axis0_request_share": sh0})
        shown.append({"rho": f"{rho:g}", "dF5 [95% CI]": f"{pct(s['mean'])} {ci(s)}",
                      "Increase / decrease / tie": idt(s), "Sign-test p": fp(s["sign_p"]),
                      "Axis-0 request share": f"{sh0:.4f}"})
        K.setdefault("rho_sweep", {})[f"{rho:g}"] = s
    W.table("t9_rho_sweep", f"Stress-weighting gain sweep: WS relative to RR (seeds {C.SEEDS_MAIN[0]}-{C.SEEDS_MAIN[-1]})",
            pd.DataFrame(rows), pd.DataFrame(shown), "rho = 6 is the main-benchmark WS run.")


def table_stiff_and_secondary(df, W, K):
    jm, jl, ks = nominal_params(0.004)
    stiff = {p: float(metric(df, "stiff", "ds4mm", p).mean()) for p in ("RR", "WS", "RA")}
    _, s_peak = paired(df, "main", "baseline", "WS", col="peak_tau_MPa")
    _, s_eff = paired(df, "main", "baseline", "WS", col="control_effort")
    ws_rmse = metric(df, "main", "baseline", "WS", "rmse_rad")
    rr_rmse = metric(df, "main", "baseline", "RR", "rmse_rad")
    abs_rmse = float((ws_rmse - rr_rmse).mean())
    raw = pd.DataFrame([{"stiff_ks_Nm_per_rad": ks, "stiff_ff_Hz": natural_frequency(jm, jl, ks),
                         "stiff_F5_RR": stiff["RR"], "stiff_F5_WS": stiff["WS"], "stiff_F5_RA": stiff["RA"],
                         "WS_dpeak_tau_mean": s_peak["mean"], "WS_dpeak_tau_lo": s_peak["ci_lo"],
                         "WS_dpeak_tau_hi": s_peak["ci_hi"], "WS_deffort_mean": s_eff["mean"],
                         "WS_deffort_lo": s_eff["ci_lo"], "WS_deffort_hi": s_eff["ci_hi"],
                         "WS_minus_RR_rmse_rad": abs_rmse}])
    shown = pd.DataFrame([
        {"Quantity": "Stiff shaft: k_s, f_f", "Value": f"{ks:.4f} N m/rad, {natural_frequency(jm, jl, ks):.3f} Hz"},
        {"Quantity": "Stiff shaft: mean F5 under RR / WS / RA (tau_ref = 5 MPa)",
         "Value": f"{stiff['RR']:.4f} / {stiff['WS']:.4f} / {stiff['RA']:.4f}"},
        {"Quantity": "Main: WS vs RR, peak surface shear stress", "Value": f"{pct(s_peak['mean'])} {ci(s_peak)}"},
        {"Quantity": "Main: WS vs RR, control effort sum(u^2) dt", "Value": f"{pct(s_eff['mean'], 3)} {ci(s_eff, 3)}"},
        {"Quantity": "Main: WS - RR tracking RMSE (absolute)", "Value": f"{abs_rmse:+.3e} rad"},
    ])
    W.table("t10_stiff_and_secondary", "Stiff transmission means and secondary WS metrics", raw, shown)
    K["stiff_means"] = stiff
    K["ws_secondary"] = {"peak_tau": s_peak, "control_effort": s_eff, "rmse_abs_rad": abs_rmse}


# ----------------------------------------------------------------------------- figures
def style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
        "axes.titlecolor": INK, "axes.labelcolor": INK2, "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "white",
        "axes.facecolor": "white", "savefig.facecolor": "white", "legend.frameon": False,
        "lines.solid_capstyle": "round",
    })


def save(fig, out, name):
    d = os.path.join(out, "figures")
    os.makedirs(d, exist_ok=True)
    fig.savefig(os.path.join(d, f"{name}.pdf"), bbox_inches="tight", metadata={"CreationDate": None})
    fig.savefig(os.path.join(d, f"{name}.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def dot_whisker(ax, labels, stats, color, seeds=None, offset=0.0, label=None):
    y = np.arange(len(labels))[::-1] + offset
    for k, s in enumerate(stats):
        if seeds is not None:
            jit = (np.random.default_rng(k).random(len(seeds[k])) - 0.5) * 0.28
            ax.scatter(seeds[k], y[k] + jit, s=9, color=AXIS, zorder=1, linewidths=0)
        ax.plot([s["ci_lo"], s["ci_hi"]], [y[k], y[k]], color=color, lw=2.0, zorder=2)
        ax.scatter([s["mean"]], [y[k]], s=34, color=color, zorder=3, edgecolors="white", linewidths=1.2,
                   label=label if k == 0 else None)
    ax.axvline(0, color=INK2, lw=0.9, zorder=0)
    ax.set_yticks(np.arange(len(labels))[::-1])
    ax.set_yticklabels(labels)
    ax.grid(axis="y", visible=False)


def figures(df, out, data_dir):
    style()
    # Fig 1: main paired effects
    pols = ["WS", "WS-B", "RA", "SO-B", "AP", "TU", "SA"]
    ds, ss = [], []
    for p in pols:
        d, s = paired(df, "main", "baseline", p)
        ds.append(d.values)
        ss.append(s)
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    dot_whisker(ax, pols, ss, S1, seeds=ds)
    ax.set_xlabel("Paired change in F5 relative to RR (%)")
    ax.set_title(f"Main benchmark: mean and 95% bootstrap interval, {C.N_SEEDS} seeds", loc="left")
    fig.text(0.0, -0.04, "Grey points are individual seeds. SO (unguarded stress-only) is off-scale "
             "and reported in Table t2.", color=MUTED, fontsize=8)
    save(fig, out, "fig1_main_paired_effects")

    # Fig 2: rho sweep
    rhos = sorted(list(C.RHO_SWEEP) + [C.RHO])
    st = []
    for rho in rhos:
        if rho == C.RHO:
            st.append(paired(df, "main", "baseline", "WS")[1])
        else:
            st.append(paired(df, "rho_sweep", f"rho{rho:g}", "WS", base_exp="main", base_cond="baseline")[1])
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    ax.plot(rhos, [s["mean"] for s in st], color=S1, lw=1.5, zorder=2)
    for rho, s in zip(rhos, st):
        ax.plot([rho, rho], [s["ci_lo"], s["ci_hi"]], color=S1, lw=2.0, zorder=2)
    ax.scatter(rhos, [s["mean"] for s in st], s=34, color=S1, zorder=3, edgecolors="white", linewidths=1.2)
    ax.axhline(0, color=INK2, lw=0.9, zorder=0)
    ax.set_xscale("log")
    ax.set_xticks(rhos)
    ax.set_xticklabels([f"{r:g}" for r in rhos])
    ax.minorticks_off()
    ax.set_xlabel("Stress-weighting gain rho (log scale)")
    ax.set_ylabel("Paired change in F5 vs RR (%)")
    ax.set_title("WS gain sweep: mean and 95% bootstrap interval", loc="left")
    save(fig, out, "fig2_rho_sweep")

    # Fig 3: robustness. Rows are aligned across the WS and RA columns; the stiff shaft
    # (a different physical configuration with its own tau_ref) gets its own row of panels
    # so its much larger effects do not compress the other conditions.
    keys = list(CONDITION_LABELS.keys())
    groups = (keys[:-1], keys[-1:])
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 4.9), sharey="row",
                             gridspec_kw={"height_ratios": [len(groups[0]), 1.6], "hspace": 0.55, "wspace": 0.08})
    for col, (pol, color) in enumerate((("WS", S1), ("RA", S2))):
        for row, ks in enumerate(groups):
            ax = axes[row, col]
            ypos = np.arange(len(ks))[::-1]
            for y, k in zip(ypos, ks):
                try:
                    s = paired(df, k[0], k[1], pol)[1]
                except KeyError:
                    ax.text(0.02, y, "not evaluated", transform=ax.get_yaxis_transform(), va="center",
                            ha="left", color=MUTED, fontsize=8)
                    continue
                ax.plot([s["ci_lo"], s["ci_hi"]], [y, y], color=color, lw=2.0, zorder=2)
                ax.scatter([s["mean"]], [y], s=34, color=color, zorder=3, edgecolors="white", linewidths=1.2)
            ax.axvline(0, color=INK2, lw=0.9, zorder=0)
            ax.set_yticks(ypos)
            ax.set_yticklabels([CONDITION_LABELS[k] for k in ks])
            ax.set_ylim(-0.6, len(ks) - 0.4)
            ax.grid(axis="y", visible=False)
            if row == 0:
                ax.set_title(f"{pol} vs RR", loc="left")
            else:
                ax.set_xlabel(f"Paired change in F5, {pol} vs RR (%)")
    st_ws = paired(df, "stiff", "ds4mm", "WS")[1]
    fig.text(0.0, -0.02, f"Mean and 95% bootstrap interval over {C.N_SEEDS} paired seeds. The stiff-shaft panels use their "
             f"own scale. Stiff shaft, WS: median {st_ws['median']:+.1f}%, {st_ws['n_increase']} of {st_ws['n']} seeds up "
             f"(sign test p = {st_ws['sign_p']:.2f}).", color=MUTED, fontsize=8)
    save(fig, out, "fig3_robustness")

    # Fig 4: per-axis effects for WS and SO-B
    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    labels_ax = ["Axis 0 (largest load)", "Axis 1", "Axis 2"]
    for pol, color, off in (("WS", S1, 0.13), ("SO-B", S2, -0.13)):
        sts = [paired(df, "main", "baseline", pol, col=f"F5_axis{i}")[1] for i in range(3)]
        dot_whisker(ax, labels_ax, sts, color, offset=off, label=pol)
    ax.set_xlabel("Paired change in per-axis F5 relative to RR (%)")
    ax.set_title("Per-axis severity change, main benchmark", loc="left")
    ax.legend(loc="lower right", handletextpad=0.3)
    save(fig, out, "fig4_per_axis")

    # Fig 5: example stress histories
    tr = {p: pd.read_csv(os.path.join(data_dir, "traces", f"main_baseline_{p}_s{C.SEEDS_MAIN[0]}.csv.gz"))
          for p in ("RR", "WS")}
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(7.0, 4.6), sharex=True,
                                  gridspec_kw={"height_ratios": [2.2, 1], "hspace": 0.35})
    w = (tr["RR"].t_s >= 3.0) & (tr["RR"].t_s <= 9.0)
    t = tr["RR"].t_s[w]
    ax.plot(t, tr["RR"].tau_MPa_axis0[w], color=S1, lw=1.3, label="RR")
    ax.plot(t, tr["WS"].tau_MPa_axis0[w], color=S2, lw=1.3, label="WS")
    ax.axhline(0, color=INK2, lw=0.9, zorder=0)
    ax.set_ylabel("Shear stress (MPa)")
    ax.set_title(f"Axis-0 shaft stress, seed {C.SEEDS_MAIN[0]}: the two histories almost coincide", loc="left")
    ax.legend(loc="upper left", ncol=2, handlelength=1.5)
    ax2.plot(t, tr["WS"].tau_MPa_axis0[w] - tr["RR"].tau_MPa_axis0[w], color=INK2, lw=1.1)
    ax2.axhline(0, color=INK2, lw=0.9, zorder=0)
    ax2.set_ylabel("WS - RR (MPa)")
    ax2.set_xlabel("Time (s)")
    ax2.set_title("Difference: largest near stress peaks", loc="left", fontsize=9)
    save(fig, out, "fig5_stress_trace")


# ----------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    df = pd.read_csv(os.path.join(a.data, "runs.csv"))
    with open(os.path.join(a.data, "calibration", "ap_shares.json")) as f:
        ap_info = json.load(f)
    os.makedirs(a.out, exist_ok=True)
    W, K = Writer(a.out), {"bootstrap": {"resamples": BOOT_RESAMPLES, "seed": BOOT_SEED, "level": 0.95}}
    K["n_evaluation_runs"] = int(len(df))
    table_main_outcomes(df, W, K)
    table_paired_effects(df, W, K)
    table_exponents(df, W, K)
    table_per_axis(df, W, K)
    table_starvation(df, W, K)
    table_ap(df, W, K, ap_info)
    table_packet_diagnostics(df, W, K, a.data)
    table_robustness(df, W, K)
    table_rho(df, W, K)
    table_stiff_and_secondary(df, W, K)
    _, s_tu_kf = paired(df, "kalman", "delayed_kf", "TU")
    K.setdefault("robustness", {}).setdefault(CONDITION_LABELS[("kalman", "delayed_kf")], {})["TU"] = s_tu_kf
    with open(os.path.join(a.out, "key_numbers.json"), "w") as f:
        json.dump(K, f, indent=2, default=float)
    with open(os.path.join(a.out, "summary.md"), "w") as f:
        f.write("# Regenerated results\n\nGenerated by `python analyze.py` from `data/runs.csv` "
                f"({len(df)} evaluation runs). Paired changes are relative to RR within seed; intervals are "
                f"95% percentile bootstrap intervals ({BOOT_RESAMPLES:,} resamples, seed {BOOT_SEED}).\n\n")
        f.write("\n".join(W.sections))
    figures(df, a.out, a.data)
    print(f"Wrote {len(W.sections)} tables, key_numbers.json, summary.md and 5 figures to {a.out}/")


if __name__ == "__main__":
    main()
