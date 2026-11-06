"""Uplink scheduling policies of manuscript Table 2.

At slot k (internal step j) the scheduler sees only controller-side information:
the current state estimates x_hat, the request steps s_i, the generation steps g_i of
the newest accepted samples and, for AP, its own allocation counts. Ages are kept as
integer numbers of internal steps, so S_i / h and (Delta_i + dt) / h are exact ratios.
Ties are broken toward the lowest axis index (numpy.argmax).
"""
from __future__ import annotations

import numpy as np

from . import config as C


def torque_variance_curve(Ad: np.ndarray, Rv: np.ndarray, ks: float, n_max: int) -> np.ndarray:
    """V(n) = c^T P_n c with P_0 = R_v and P_{n+1} = A_d P_n A_d^T (open-loop growth from R_v)."""
    c = ks * np.array([1.0, 0.0, -1.0, 0.0])
    V = np.empty(n_max + 1)
    P = Rv.copy()
    for n in range(n_max + 1):
        V[n] = c @ P @ c
        P = Ad @ P @ Ad.T
    return V


class Scheduler:
    def __init__(self, policy: str, slot_steps: int, rho: float, ks_nom: float, t_ref: float,
                 ap_shares=None, tu_curve: np.ndarray | None = None):
        if policy not in C.POLICIES:
            raise ValueError(f"unknown policy {policy}")
        self.policy = policy
        self.h = float(slot_steps)
        self.guard = C.GUARD_SLOTS * slot_steps
        self.rho = rho
        self.ks = ks_nom
        self.t_ref = t_ref
        if policy == "AP":
            if ap_shares is None:
                raise ValueError("AP needs calibrated shares")
            self.p = np.asarray(ap_shares, float)
        if policy == "TU":
            if tu_curve is None:
                raise ValueError("TU needs the torque variance curve")
            self.V = tu_curve

    def stress_weight(self, xh: np.ndarray) -> np.ndarray:
        """R_i = 1 + rho (|T_hat_i| / T_ref)^3 with T_hat_i = k_s (q_m_hat - q_l_hat) (Eq. 12)."""
        t_hat = self.ks * (xh[:, 0] - xh[:, 2])
        return 1.0 + self.rho * (np.abs(t_hat) / self.t_ref) ** 3

    def choose(self, k: int, j: int, s_req: np.ndarray, g: np.ndarray, xh: np.ndarray,
               n_alloc: np.ndarray) -> int:
        pol = self.policy
        if pol == "RR":
            return k % C.N_AXES
        S = (j - s_req).astype(float)          # S_i in steps
        if pol == "SA":
            return int(np.argmax((S / self.h) ** 2))
        if pol == "RA":
            age = (j - g).astype(float)           # Delta_i in steps
            return int(np.argmax(((age + 1.0) / self.h) ** 2))
        if pol in ("WS-B", "SO-B") and S.max() >= self.guard:
            return int(np.argmax(S))
        if pol in ("SO", "SO-B"):
            return int(np.argmax(self.stress_weight(xh)))
        if pol in ("WS", "WS-B"):
            return int(np.argmax((S / self.h) ** 2 * self.stress_weight(xh)))
        if pol == "AP":
            return int(np.argmax((k + 1) * self.p - n_alloc))
        if pol == "TU":
            return int(np.argmax(self.V[(j - s_req)]))
        raise AssertionError(pol)
