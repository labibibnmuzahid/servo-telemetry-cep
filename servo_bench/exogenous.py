"""Pre-generated exogenous processes (manuscript Sections 3.3-3.4).

All random draws are made before the closed loop runs, from six independent
streams spawned from one `numpy.random.SeedSequence(seed)`:

    0 parameter perturbations   3 colored disturbance noise
    1 reference phases          4 measurement noise (per slot and axis)
    2 load pulses               5 channel loss and delay (per slot)

Arrays are indexed by time step or by slot only, so a scheduler's choices can
never change the realisation: every policy run with the same seed and
environment sees identical disturbances, noise and channel outcomes.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import lfilter

from . import config as C


@dataclass
class Exogenous:
    factors: np.ndarray      # (3, 3) multiplicative factors for (J_m, J_l, k_s) per axis
    phases: np.ndarray       # (3,) reference phase offsets [rad]
    ref: np.ndarray          # (N, 3) reference r_i(t_j) [rad]
    dist: np.ndarray         # (N, 3) load disturbance d_L,i(t_j) [N m], held over each step
    pulses: list             # (axis, t0, duration, signed amplitude) for documentation
    meas_noise: np.ndarray   # (n_slots, 3, 4) sensor noise v for a packet sampled at slot k from axis i
    lost: np.ndarray         # (n_slots,) bool, packet sent in slot k is lost
    delay: np.ndarray        # (n_slots,) int, delivery delay in steps for slot k
    congested: np.ndarray    # (n_slots,) bool


def streams(seed: int) -> list[np.random.Generator]:
    return [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(6)]


def generate(cfg) -> Exogenous:
    rng_par, rng_phase, rng_pulse, rng_col, rng_meas, rng_chan = streams(cfg.seed)
    n = cfg.n_steps
    t = np.arange(n) * C.DT

    # parameter perturbation: one unit draw per (axis, parameter), scaled by the mismatch level,
    # so the 0%, 12% and 25% conditions share the same underlying draws.
    unit = rng_par.uniform(-1.0, 1.0, size=(C.N_AXES, 3))
    factors = 1.0 + cfg.mismatch * unit

    phases = rng_phase.uniform(-C.REF_PHASE_HALFWIDTH, C.REF_PHASE_HALFWIDTH, size=C.N_AXES)
    amp = np.asarray(C.REF_AMP)
    freq = np.asarray(C.REF_FREQ)
    ramp = 1.0 - np.exp(-t / C.REF_RAMP_TAU)
    ref = amp[None, :] * np.sin(2 * np.pi * freq[None, :] * t[:, None] + phases[None, :]) * ramp[:, None]

    # load pulses: nominal onsets 1 + 0.37 i + 2.4 n, jittered, sin^2 profile, random sign
    n_rows = int(cfg.t_run / C.PULSE_PERIOD) + 2
    jitter = rng_pulse.uniform(-C.PULSE_JITTER, C.PULSE_JITTER, size=(n_rows, C.N_AXES))
    dur = rng_pulse.uniform(*C.PULSE_DURATION, size=(n_rows, C.N_AXES))
    sign = np.where(rng_pulse.random(size=(n_rows, C.N_AXES)) < 0.5, -1.0, 1.0)
    dist = np.zeros((n, C.N_AXES))
    pulses = []
    for i in range(C.N_AXES):
        for r in range(n_rows):
            t0 = C.PULSE_FIRST + C.PULSE_AXIS_OFFSET * i + C.PULSE_PERIOD * r + jitter[r, i]
            if t0 >= cfg.t_run:
                continue
            d = dur[r, i]
            a = sign[r, i] * C.PULSE_AMP[i]
            mask = (t >= t0) & (t < t0 + d)
            dist[mask, i] += a * np.sin(np.pi * (t[mask] - t0) / d) ** 2
            pulses.append((i, float(t0), float(d), float(a)))

    # colored disturbance 0.002 z_j, z_j = 0.97 z_{j-1} + 0.03 w_j, z_{-1} = 0
    w = rng_col.standard_normal(size=(n, C.N_AXES))
    z = lfilter([C.COLORED_INPUT], [1.0, -C.COLORED_POLE], w, axis=0)
    dist += C.COLORED_GAIN * z

    # sensor noise for every (slot, axis), drawn whether or not that axis is scheduled
    n_slots = cfg.n_slots
    meas_noise = rng_meas.standard_normal(size=(n_slots, C.N_AXES, 4)) * np.asarray(C.SIGMA)

    # channel outcome for every slot
    u_loss = rng_chan.random(size=n_slots)
    d_base = rng_chan.integers(C.DELAY_BASE[0], C.DELAY_BASE[1] + 1, size=n_slots)
    d_cong = rng_chan.integers(C.DELAY_CONG[0], C.DELAY_CONG[1] + 1, size=n_slots)
    slot_step = np.arange(n_slots) * cfg.slot_steps
    phase = slot_step % C.CONG_PERIOD_STEPS
    congested = (phase > C.CONG_START_STEPS) & (phase < C.CONG_END_STEPS)
    lost = u_loss < np.where(congested, C.LOSS_CONG, C.LOSS_BASE)
    delay = np.where(congested, d_cong, d_base).astype(np.int64) * int(cfg.delay_factor)

    return Exogenous(factors=factors, phases=phases, ref=ref, dist=dist, pulses=pulses,
                     meas_noise=meas_noise, lost=lost, delay=delay, congested=congested)
