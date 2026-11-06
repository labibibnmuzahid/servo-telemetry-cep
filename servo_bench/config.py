"""Nominal benchmark parameters, scheduling policies and the experiment matrix.

Every constant below is taken from the manuscript (Sections 3.2-3.8, Tables 1-3).
Where the manuscript leaves a detail open, the choice made here is marked
"IMPLEMENTATION CHOICE" and is listed in README.md.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# ---------------------------------------------------------------- timing
DT = 0.002                 # internal simulation / control period [s]
T_RUN = 16.0               # nominal trial duration [s]
SLOT_MS = 10.0             # nominal telemetry slot interval h [ms]
N_AXES = 3

# ---------------------------------------------------------------- geometry and material
G_SHEAR = 79e9             # shear modulus [Pa]
DISK_RADIUS = 0.09         # solid steel disks [m]
MASS_MOTOR = 0.8           # [kg]
MASS_LOAD = 1.4            # [kg]
SHAFT_DIAMETER = 0.002     # d_s [m]
SHAFT_LENGTH = 0.20        # L_s [m]
C_S = 0.004                # internal torsional damping [N m s/rad]
B_M = 0.0005               # motor viscous friction [N m s/rad]
B_L = 0.0005               # load viscous friction [N m s/rad]
PARAM_MISMATCH = 0.12      # uniform +/- perturbation of (J_m, J_l, k_s)

# ---------------------------------------------------------------- control
U_MAX = 0.30               # motor torque saturation [N m]
Q_LQR = (2.0, 0.003, 25.0, 0.015)
R_LQR = 1.0

# ---------------------------------------------------------------- stress normalisation and weighting
TAU_REF = 40e6             # reference shear stress [Pa]
RHO = 6.0                  # cubic stress-weighting gain
GUARD_SLOTS = 6            # S_guard = 6 h

# ---------------------------------------------------------------- motions and disturbances
REF_AMP = (0.60, 0.35, 0.22)       # [rad]
REF_FREQ = (0.45, 0.27, 0.18)      # [Hz]
REF_PHASE_HALFWIDTH = 0.2          # phi_i ~ U[-0.2, 0.2] rad
REF_RAMP_TAU = 0.4                 # (1 - exp(-t/0.4))
PULSE_AMP = (0.025, 0.014, 0.009)  # A_i [N m]
PULSE_FIRST = 1.0                  # t0 = 1 + 0.37 i + 2.4 n
PULSE_AXIS_OFFSET = 0.37
PULSE_PERIOD = 2.4
PULSE_JITTER = 0.15                # U[-0.15, 0.15] s
PULSE_DURATION = (0.25, 0.55)      # D ~ U[0.25, 0.55] s
COLORED_GAIN = 0.002               # 0.002 z_j [N m]
COLORED_POLE = 0.97                # z_j = 0.97 z_{j-1} + 0.03 w_j
COLORED_INPUT = 0.03

# ---------------------------------------------------------------- sensing
SIGMA = (1e-4, 0.003, 1e-4, 0.003)  # SD of (q_m, w_m, q_l, w_l) noise

# ---------------------------------------------------------------- channel
DELAY_BASE = (2, 8)        # uniform integer steps, inclusive
LOSS_BASE = 0.02
DELAY_CONG = (15, 30)      # uniform integer steps, inclusive
LOSS_CONG = 0.15
# congestion when 2.2 < (t mod 3.2) < 2.7 s; evaluated in integer steps
CONG_PERIOD_STEPS = 1600
CONG_START_STEPS = 1100
CONG_END_STEPS = 1350

# ---------------------------------------------------------------- delayed Kalman filter (sensitivity)
KF_PULSE_FACTOR = 0.4      # Q_c,i = 0.4 A_i^2 E E^T + 1e-5 B B^T
KF_INPUT_NOISE = 1e-5

POLICIES = ("RR", "SA", "RA", "SO", "WS", "WS-B", "SO-B", "AP", "TU")
RHO_SWEEP = (0.5, 1.5, 3.0, 12.0, 24.0, 48.0)

N_SEEDS = 96               # paired seeds per experimental cell (version 2; version 1 used 24)
SEEDS_MAIN = tuple(range(3201, 3201 + N_SEEDS))
SEEDS_OPERATING = tuple(range(4201, 4201 + N_SEEDS))
SEEDS_LONG = tuple(range(5201, 5201 + N_SEEDS))
SEEDS_STIFF = tuple(range(6201, 6201 + N_SEEDS))
SEEDS_CALIBRATION = tuple(range(2201, 2201 + N_SEEDS))


@dataclass(frozen=True)
class RunConfig:
    """One closed-loop trial. Runs with equal `seed` and environment share all exogenous draws."""

    experiment: str
    condition: str
    policy: str
    seed: int
    rho: float = RHO
    t_run: float = T_RUN
    slot_ms: float = SLOT_MS
    delay_factor: int = 1
    mismatch: float = PARAM_MISMATCH
    shaft_diameter: float = SHAFT_DIAMETER
    tau_ref: float = TAU_REF
    estimator: str = "reset"            # "reset" (state-reset predictor) or "kf" (delayed Kalman filter)
    ap_shares: tuple | None = field(default=None, compare=False)

    @property
    def slot_steps(self) -> int:
        return int(round(self.slot_ms / 1000.0 / DT))

    @property
    def n_steps(self) -> int:
        return int(round(self.t_run / DT))

    @property
    def n_slots(self) -> int:
        return -(-self.n_steps // self.slot_steps)   # ceil: slots at steps 0, h, 2h, ... < n_steps

    @property
    def run_id(self) -> str:
        return f"{self.experiment}__{self.condition}__{self.policy}__s{self.seed}"


def _cfg(**kw) -> RunConfig:
    return RunConfig(**kw)


def calibration_configs() -> list[RunConfig]:
    """WS runs on independent calibration seeds used to derive the fixed AP target shares."""
    return [_cfg(experiment="calibration", condition="baseline", policy="WS", seed=s)
            for s in SEEDS_CALIBRATION]


def evaluation_configs(ap_shares: tuple) -> list[RunConfig]:
    """The evaluation runs of manuscript Table 3 (38 cells x N_SEEDS seeds)."""
    runs: list[RunConfig] = []
    # Main benchmark: 9 policies
    for p in POLICIES:
        for s in SEEDS_MAIN:
            runs.append(_cfg(experiment="main", condition="baseline", policy=p, seed=s,
                             ap_shares=ap_shares if p == "AP" else None))
    # Operating conditions: RR, RA, WS x 5 conditions
    conditions = {
        "slot6ms": dict(slot_ms=6.0),
        "slot20ms": dict(slot_ms=20.0),
        "delay2x": dict(delay_factor=2),
        "mismatch0": dict(mismatch=0.0),
        "mismatch25": dict(mismatch=0.25),
    }
    for cond, kw in conditions.items():
        for p in ("RR", "RA", "WS"):
            for s in SEEDS_OPERATING:
                runs.append(_cfg(experiment="operating", condition=cond, policy=p, seed=s, **kw))
    # Stress-weight sweep: WS at 6 gains (RR baseline reused from main)
    for rho in RHO_SWEEP:
        for s in SEEDS_MAIN:
            runs.append(_cfg(experiment="rho_sweep", condition=f"rho{rho:g}", policy="WS", seed=s, rho=rho))
    # Long horizon: RR, WS at 64 s
    for p in ("RR", "WS"):
        for s in SEEDS_LONG:
            runs.append(_cfg(experiment="long_horizon", condition="t64s", policy=p, seed=s, t_run=64.0))
    # Stiff transmission: RR, RA, WS with d_s = 4 mm, tau_ref = 5 MPa
    for p in ("RR", "RA", "WS"):
        for s in SEEDS_STIFF:
            runs.append(_cfg(experiment="stiff", condition="ds4mm", policy=p, seed=s,
                             shaft_diameter=0.004, tau_ref=5e6))
    # Estimator robustness: RR, WS, TU with the delayed Kalman filter
    for p in ("RR", "WS", "TU"):
        for s in SEEDS_MAIN:
            runs.append(_cfg(experiment="kalman", condition="delayed_kf", policy=p, seed=s, estimator="kf"))
    return runs
