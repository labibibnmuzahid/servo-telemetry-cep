"""Controller-side state estimators (manuscript Section 3.5).

Between arrivals both estimators propagate the nominal model under the applied
saturated inputs (done in the simulator). They differ in what happens when a packet
y_s sampled at step s is accepted at step j:

* ResetPredictor: x_hat_j = A_d^(j-s) y_s + sum_l A_d^(j-s-1-l) B_d u_(s+l)   (Eq. 9)
* DelayedKF: an in-order Kalman filter that keeps the posterior at the generation
  step of the last accepted sample, propagates mean and covariance to s, applies the
  measurement update, then replays the known inputs from s to j.
"""
from __future__ import annotations

import numpy as np


def replay(x0: np.ndarray, Ad: np.ndarray, Bd: np.ndarray, u_axis: np.ndarray, start: int, stop: int) -> np.ndarray:
    """Propagate x0 from step `start` to step `stop` under the recorded inputs u_axis[start:stop]."""
    x = x0.copy()
    for l in range(start, stop):
        x = Ad @ x + Bd * u_axis[l]
    return x


class ResetPredictor:
    name = "reset"

    def __init__(self, Ad: np.ndarray, Bd: np.ndarray):
        self.Ad, self.Bd = Ad, Bd

    def accept(self, axis: int, gen: int, y: np.ndarray, j: int, u_hist: np.ndarray) -> np.ndarray:
        return replay(y, self.Ad, self.Bd, u_hist[:, axis], gen, j)


class DelayedKF:
    name = "kf"

    def __init__(self, Ad: np.ndarray, Bd: np.ndarray, Qd: np.ndarray, Rv: np.ndarray, n_axes: int = 3):
        """Qd: (n_axes, 4, 4) per-axis discrete process covariance; Rv: (4, 4) measurement covariance.

        IMPLEMENTATION CHOICE: the initial posterior is mean 0 (the plant starts at rest)
        with covariance R_v at step 0.
        """
        self.Ad, self.Bd, self.Qd, self.Rv = Ad, Bd, Qd, Rv
        self.m = np.zeros((n_axes, 4))
        self.P = np.repeat(Rv[None, :, :], n_axes, axis=0)
        self.g = np.zeros(n_axes, dtype=np.int64)
        self.I = np.eye(4)

    def accept(self, axis: int, gen: int, y: np.ndarray, j: int, u_hist: np.ndarray) -> np.ndarray:
        Ad, Bd, Q = self.Ad, self.Bd, self.Qd[axis]
        m = self.m[axis].copy()
        P = self.P[axis].copy()
        u = u_hist[:, axis]
        for l in range(int(self.g[axis]), gen):
            m = Ad @ m + Bd * u[l]
            P = Ad @ P @ Ad.T + Q
        S = P + self.Rv
        K = np.linalg.solve(S, P).T                     # P S^-1 (P, S symmetric)
        m = m + K @ (y - m)
        IK = self.I - K
        P = IK @ P @ IK.T + K @ self.Rv @ K.T           # Joseph form
        P = 0.5 * (P + P.T)
        self.m[axis], self.P[axis], self.g[axis] = m, P, gen
        return replay(m, Ad, Bd, u, gen, j)
