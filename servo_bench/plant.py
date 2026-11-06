"""Two-inertia flexible drive model, exact ZOH discretisation, Van Loan noise
discretisation and the continuous-time LQR (manuscript Section 3.2 and 3.5)."""
from __future__ import annotations

import numpy as np
import scipy.linalg as sla

from . import config as C


def nominal_params(shaft_diameter: float = C.SHAFT_DIAMETER) -> tuple[float, float, float]:
    """Nominal (J_m, J_l, k_s) from the disk and shaft geometry."""
    jm = 0.5 * C.MASS_MOTOR * C.DISK_RADIUS ** 2
    jl = 0.5 * C.MASS_LOAD * C.DISK_RADIUS ** 2
    ks = C.G_SHEAR * np.pi * shaft_diameter ** 4 / (32.0 * C.SHAFT_LENGTH)
    return jm, jl, ks


def natural_frequency(jm: float, jl: float, ks: float) -> float:
    """Undamped torsional natural frequency f_f [Hz] (Eq. 6)."""
    return float(np.sqrt(ks * (1.0 / jm + 1.0 / jl)) / (2.0 * np.pi))


def continuous_matrices(jm: float, jl: float, ks: float, cs: float = C.C_S,
                        bm: float = C.B_M, bl: float = C.B_L):
    """State x = [q_m, w_m, q_l, w_l]; returns A, B (input u), E (disturbance d_L) of Eqs. (1)-(4)."""
    A = np.array([
        [0.0, 1.0, 0.0, 0.0],
        [-ks / jm, -(cs + bm) / jm, ks / jm, cs / jm],
        [0.0, 0.0, 0.0, 1.0],
        [ks / jl, cs / jl, -ks / jl, -(cs + bl) / jl],
    ])
    B = np.array([0.0, 1.0 / jm, 0.0, 0.0])
    E = np.array([0.0, 0.0, 0.0, -1.0 / jl])
    return A, B, E


def zoh(A: np.ndarray, inputs: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Exact zero-order-hold discretisation via one augmented matrix exponential.

    inputs: (n, m) input matrix; returns A_d = exp(A dt) and the (n, m) matrix
    int_0^dt exp(A eta) inputs d eta.
    """
    inputs = np.atleast_2d(np.asarray(inputs, float))
    if inputs.shape[0] != A.shape[0]:
        inputs = inputs.T
    n, m = inputs.shape
    M = np.zeros((n + m, n + m))
    M[:n, :n] = A
    M[:n, n:] = inputs
    Md = sla.expm(M * dt)
    return Md[:n, :n], Md[:n, n:]


def van_loan(A: np.ndarray, Qc: np.ndarray, dt: float) -> np.ndarray:
    """Discrete process covariance Q_d = int_0^dt exp(A s) Qc exp(A^T s) ds (Van Loan, 1978)."""
    n = A.shape[0]
    M = np.zeros((2 * n, 2 * n))
    M[:n, :n] = -A
    M[:n, n:] = Qc
    M[n:, n:] = A.T
    Md = sla.expm(M * dt)
    Qd = Md[n:, n:].T @ Md[:n, n:]
    return 0.5 * (Qd + Qd.T)


def lqr_gain(A: np.ndarray, B: np.ndarray, Q=C.Q_LQR, R=C.R_LQR) -> np.ndarray:
    """Continuous-time LQR gain K (u = -K x) from the algebraic Riccati equation."""
    Bm = B.reshape(-1, 1)
    Qm = np.diag(Q)
    Rm = np.array([[R]], float)
    P = sla.solve_continuous_are(A, Bm, Qm, Rm)
    return np.linalg.solve(Rm, Bm.T @ P).ravel()


def shear_stress(torque, shaft_diameter: float):
    """Surface shear stress tau = 16 T / (pi d^3) [Pa] (Eq. 15)."""
    return 16.0 * np.asarray(torque) / (np.pi * shaft_diameter ** 3)


def reference_torque(tau_ref: float, shaft_diameter: float) -> float:
    """T_ref = tau_ref pi d^3 / 16 [N m] (Eq. 14)."""
    return tau_ref * np.pi * shaft_diameter ** 3 / 16.0


def kf_process_covariance(axis: int, jm: float, jl: float, ks: float, dt: float = C.DT) -> np.ndarray:
    """Discretised Q_c,i = 0.4 A_i^2 E E^T + 1e-5 B B^T (Eq. 12) for the nominal model of one axis."""
    A, B, E = continuous_matrices(jm, jl, ks)
    Qc = C.KF_PULSE_FACTOR * C.PULSE_AMP[axis] ** 2 * np.outer(E, E) + C.KF_INPUT_NOISE * np.outer(B, B)
    return van_loan(A, Qc, dt)
