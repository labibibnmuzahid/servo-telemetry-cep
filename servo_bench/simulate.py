"""Closed-loop simulation of one paired trial (manuscript Sections 3.1-3.7).

Step ordering inside internal step j (t_j = j dt):
  1. deliver packets whose arrival step is j, oldest generation first; a packet is
     accepted only if it is newer than the newest accepted sample of its axis,
     otherwise it is discarded as stale;
  2. if j is a slot step (j = k h), the scheduler picks one axis from controller-side
     information; that axis samples y = x_j + v and the packet is lost or delayed
     according to the pre-generated channel draws of slot k;
  3. every axis computes u_j = sat_0.30{-K (x_hat_j - [r_j, 0, r_j, 0])};
  4. the true plants advance over [t_j, t_j+1) under u_j and d_L(t_j) (exact ZOH),
     and the estimates advance with the nominal model under the same u_j.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import config as C
from .estimators import DelayedKF, ResetPredictor
from .exogenous import Exogenous, generate
from .plant import (continuous_matrices, kf_process_covariance, lqr_gain, nominal_params,
                    reference_torque, shear_stress, zoh)
from .rainflow import rainflow, severity
from .schedulers import Scheduler, torque_variance_curve

STATUS_NAMES = {0: "pending", 1: "accepted", 2: "stale", 3: "lost"}
EXPONENTS = (3, 5, 7)


@dataclass
class Model:
    nominal: tuple          # (J_m, J_l, k_s) nominal
    Ad_n: np.ndarray        # nominal ZOH model used by controller and estimator
    Bd_n: np.ndarray
    K: np.ndarray           # LQR gain
    A_true: np.ndarray      # (3, 4, 4) continuous true plants
    B_true: np.ndarray      # (3, 4)
    E_true: np.ndarray      # (3, 4)
    Ad_true: np.ndarray     # (3, 4, 4)
    Bd_true: np.ndarray     # (3, 4)
    Ed_true: np.ndarray     # (3, 4)
    ks_true: np.ndarray     # (3,)
    Qd_kf: np.ndarray       # (3, 4, 4) delayed-KF process covariance


def build_model(cfg, ex: Exogenous) -> Model:
    jm0, jl0, ks0 = nominal_params(cfg.shaft_diameter)
    A_n, B_n, _ = continuous_matrices(jm0, jl0, ks0)
    Ad_n, Bd_n = zoh(A_n, B_n[:, None], C.DT)
    K = lqr_gain(A_n, B_n)
    A_t = np.empty((C.N_AXES, 4, 4)); B_t = np.empty((C.N_AXES, 4)); E_t = np.empty((C.N_AXES, 4))
    Ad_t = np.empty((C.N_AXES, 4, 4)); Bd_t = np.empty((C.N_AXES, 4)); Ed_t = np.empty((C.N_AXES, 4))
    ks_t = np.empty(C.N_AXES)
    for i in range(C.N_AXES):
        fjm, fjl, fks = ex.factors[i]
        A, B, E = continuous_matrices(jm0 * fjm, jl0 * fjl, ks0 * fks)
        Ad, BEd = zoh(A, np.column_stack([B, E]), C.DT)
        A_t[i], B_t[i], E_t[i] = A, B, E
        Ad_t[i], Bd_t[i], Ed_t[i] = Ad, BEd[:, 0], BEd[:, 1]
        ks_t[i] = ks0 * fks
    Qd = np.stack([kf_process_covariance(i, jm0, jl0, ks0) for i in range(C.N_AXES)])
    return Model((jm0, jl0, ks0), Ad_n, Bd_n[:, 0], K, A_t, B_t, E_t, Ad_t, Bd_t, Ed_t, ks_t, Qd)


def simulate(cfg, return_traces: bool = False) -> dict:
    ex = generate(cfg)
    mdl = build_model(cfg, ex)
    N, hs, n_slots = cfg.n_steps, cfg.slot_steps, cfg.n_slots
    jm0, jl0, ks0 = mdl.nominal
    Rv = np.diag(np.square(C.SIGMA))
    t_ref = reference_torque(cfg.tau_ref, cfg.shaft_diameter)
    tu_curve = torque_variance_curve(mdl.Ad_n, Rv, ks0, N + 1) if cfg.policy == "TU" else None
    sched = Scheduler(cfg.policy, hs, cfg.rho, ks0, t_ref, cfg.ap_shares, tu_curve)
    if cfg.estimator == "reset":
        est = ResetPredictor(mdl.Ad_n, mdl.Bd_n)
    elif cfg.estimator == "kf":
        est = DelayedKF(mdl.Ad_n, mdl.Bd_n, mdl.Qd_kf, Rv)
    else:
        raise ValueError(cfg.estimator)

    K0, K1, K2, K3 = mdl.K
    umax = C.U_MAX
    ref, dist = ex.ref, ex.dist
    Ad_t, Bd_t, Ed_t, ks_t = mdl.Ad_true, mdl.Bd_true, mdl.Ed_true, mdl.ks_true
    Ad_nT, Bd_n = mdl.Ad_n.T, mdl.Bd_n

    x = np.zeros((C.N_AXES, 4))
    xh = np.zeros((C.N_AXES, 4))
    u_hist = np.zeros((N, C.N_AXES))
    twist = np.empty((N, C.N_AXES))
    q_l = np.empty((N, C.N_AXES))
    age = np.empty((N, C.N_AXES), dtype=np.int64)
    xs = np.empty((N + 1, C.N_AXES, 4)) if return_traces else None
    xhs = np.empty((N, C.N_AXES, 4)) if return_traces else None

    s_req = np.full(C.N_AXES, -1, dtype=np.int64)      # s_i = -dt
    g = np.zeros(C.N_AXES, dtype=np.int64)             # g_i = 0
    last_acc = np.full(C.N_AXES, -1, dtype=np.int64)
    n_alloc = np.zeros(C.N_AXES, dtype=np.int64)
    arrivals: list[list[int]] = [[] for _ in range(N)]

    pk_axis = np.full(n_slots, -1, dtype=np.int64)
    pk_status = np.zeros(n_slots, dtype=np.int8)
    pk_arr = np.full(n_slots, -1, dtype=np.int64)
    pk_y = np.zeros((n_slots, 4))
    pk_T = np.full(n_slots, np.nan)        # true elastic torque at arrival step
    pk_Tb = np.full(n_slots, np.nan)       # estimated torque just before acceptance
    pk_Ta = np.full(n_slots, np.nan)       # estimated torque just after acceptance
    pk_du = np.full(n_slots, np.nan)       # change of saturated command caused by acceptance

    def command(xa: np.ndarray, r: float) -> float:
        v = -(K0 * (xa[0] - r) + K1 * xa[1] + K2 * (xa[2] - r) + K3 * xa[3])
        return min(max(v, -umax), umax)

    for j in range(N):
        due = arrivals[j]
        if due:
            for k in sorted(due):
                a = pk_axis[k]
                gen = k * hs
                pk_T[k] = ks_t[a] * (x[a, 0] - x[a, 2])
                if gen > last_acc[a]:
                    r = ref[j, a]
                    tb = ks0 * (xh[a, 0] - xh[a, 2])
                    ub = command(xh[a], r)
                    xh[a] = est.accept(a, gen, pk_y[k], j, u_hist)
                    pk_Tb[k] = tb
                    pk_Ta[k] = ks0 * (xh[a, 0] - xh[a, 2])
                    pk_du[k] = command(xh[a], r) - ub
                    pk_status[k] = 1
                    last_acc[a] = gen
                    g[a] = gen
                else:
                    pk_status[k] = 2
        if j % hs == 0:
            k = j // hs
            a = sched.choose(k, j, s_req, g, xh, n_alloc)
            n_alloc[a] += 1
            s_req[a] = j
            pk_axis[k] = a
            pk_y[k] = x[a] + ex.meas_noise[k, a]
            if ex.lost[k]:
                pk_status[k] = 3
            else:
                arr = j + int(ex.delay[k])
                pk_arr[k] = arr
                if arr < N:
                    arrivals[arr].append(k)
        r = ref[j]
        u = -(K0 * (xh[:, 0] - r) + K1 * xh[:, 1] + K2 * (xh[:, 2] - r) + K3 * xh[:, 3])
        np.clip(u, -umax, umax, out=u)
        u_hist[j] = u
        twist[j] = x[:, 0] - x[:, 2]
        q_l[j] = x[:, 2]
        age[j] = j - g
        if return_traces:
            xs[j] = x
            xhs[j] = xh
        x = np.einsum("aij,aj->ai", Ad_t, x) + Bd_t * u[:, None] + Ed_t * dist[j][:, None]
        xh = xh @ Ad_nT + u[:, None] * Bd_n[None, :]
    if return_traces:
        xs[N] = x

    # ------------------------------------------------------------ mechanical evaluation
    torque = twist * ks_t[None, :]
    tau = shear_stress(torque, cfg.shaft_diameter)
    summary: dict = {}
    for m in EXPONENTS:
        summary[f"F{m}"] = 0.0
    n_cycles = 0
    for i in range(C.N_AXES):
        rng_, cnt = rainflow(tau[:, i])
        n_cycles += int(cnt.size)
        for m in EXPONENTS:
            f = severity(rng_, cnt, cfg.tau_ref, m)
            summary[f"F{m}_axis{i}"] = f
            summary[f"F{m}"] += f
    summary["rainflow_records"] = n_cycles
    err = ref - q_l
    summary["rmse_rad"] = float(np.sqrt(np.mean(err ** 2)))
    for i in range(C.N_AXES):
        summary[f"rmse_rad_axis{i}"] = float(np.sqrt(np.mean(err[:, i] ** 2)))
    peak = np.max(np.abs(tau), axis=0) / 1e6
    summary["peak_tau_MPa"] = float(peak.max())
    for i in range(C.N_AXES):
        summary[f"peak_tau_MPa_axis{i}"] = float(peak[i])
    summary["control_effort"] = float(np.sum(u_hist ** 2) * C.DT)
    age_ms = age * (C.DT * 1e3)
    summary["mean_age_ms"] = float(age_ms.mean())
    summary["p99_age_ms"] = float(np.percentile(age_ms, 99))
    for i in range(C.N_AXES):
        summary[f"max_age_ms_axis{i}"] = float(age_ms[:, i].max())

    # ------------------------------------------------------------ packet accounting
    for name, code in (("accepted", 1), ("stale", 2), ("lost", 3), ("pending", 0)):
        tot = 0
        for i in range(C.N_AXES):
            c = int(np.sum((pk_axis == i) & (pk_status == code)))
            summary[f"{name}_axis{i}"] = c
            tot += c
        summary[name] = tot
    for i in range(C.N_AXES):
        summary[f"attempted_axis{i}"] = int(n_alloc[i])
    summary["attempted"] = int(n_alloc.sum())
    acc = pk_status == 1
    aligned = acc & (pk_du * pk_T > 0)
    summary["aligned_corrections"] = int(aligned.sum())

    packets = {
        "slot": np.arange(n_slots),
        "axis": pk_axis,
        "gen_step": np.arange(n_slots) * hs,
        "arr_step": pk_arr,
        "delay_steps": np.where(pk_status == 3, -1, ex.delay),
        "congested": ex.congested.astype(np.int8),
        "status": pk_status,
        "torque_true_Nm": pk_T,
        "torque_est_before_Nm": pk_Tb,
        "torque_est_after_Nm": pk_Ta,
        "du_cmd_Nm": pk_du,
        "aligned": np.where(acc, aligned.astype(np.int8), -1),
    }
    out = {"summary": summary, "packets": packets}
    if return_traces:
        out["traces"] = {"x": xs, "x_hat": xhs, "u": u_hist, "dist": dist, "ref": ref,
                         "tau": tau, "model": mdl, "exogenous": ex}
    return out
