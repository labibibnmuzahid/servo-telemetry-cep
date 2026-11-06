# Replication package: Stress-weighted telemetry and stress-cycle severity in flexible servo axes

Abdullah Al Junayed (Mahin) and Labib Ibn Muzahid
International University of Business Agriculture and Technology (IUBAT), Dhaka, Bangladesh
Contact: junayed.mahin@iubat.education (corresponding), labib@iubat.education

This package contains the complete simulation code, the archived results of all 936 simulation
runs (912 evaluation runs and 24 calibration runs), per-packet telemetry logs, the 114-check
verification suite, and the scripts that regenerate every table, statistic and figure for the
article *Stress-Weighted Telemetry and Stress-Cycle Severity in Flexible Servo Axes: A
Reproducible Computational Study* (submitted to Mechanical Systems and Signal Processing).

**Dataset DOI:** 10.17632/hbcy5m7rrk.1 (Mendeley Data)
**Licence:** CC BY 4.0 (see `LICENSE`)

The study asks whether scheduling a shared, lossy sensor uplink by predicted elastic shaft torque
reduces fatigue-relevant stress cycles in three flexible two-inertia servo axes. Stress cycles are
counted offline with three-point rainflow counting on the true shaft stress and summarised by the
normalised severity indices F3, F5 and F7.

---

## 1. Quick start

Requirements: Python 3.11 or newer with NumPy, SciPy, pandas and Matplotlib
(`pip install -r requirements.txt`). No other software is needed.

```bash
python run_experiments.py         # 24 calibration + 912 evaluation runs -> data/      (~100 s, 2 workers)
python validate.py                # 114 verification checks -> results/validation_report.txt   (~15 s)
python check_stress_sampling.py   # 0.5 ms observation-grid check -> results/stress_sampling_check.*  (~5 s)
python analyze.py                 # tables, statistics, figures -> results/                  (~3 s)
```

The archived `data/` and `results/` folders were produced by exactly these commands, so
`validate.py` and `analyze.py` can be run directly on the archive without re-running the
simulations. `run_experiments.py --workers N` sets the number of worker processes.

## 2. Contents

```
README.md                  this file
LICENSE                    CC BY 4.0
CITATION.cff               citation metadata
requirements.txt           Python packages and the versions used to build the archive
run_experiments.py         runs the 24 AP-calibration runs and the 912 evaluation runs
validate.py                114 verification checks
check_stress_sampling.py   2 ms vs 0.5 ms stress-reconstruction check
analyze.py                 regenerates all tables, statistics and figures
servo_bench/               simulation package
    config.py              every parameter, the policy list and the experiment matrix
    plant.py               two-inertia model, exact ZOH, Van Loan discretisation, LQR
    exogenous.py           seeded reference, disturbance, sensor-noise and channel draws
    schedulers.py          the nine scheduling policies
    estimators.py          state-reset predictor and in-order delayed Kalman filter
    simulate.py            closed-loop simulation of one trial and its metrics
    rainflow.py            three-point rainflow counting and the severity index
    stats.py               paired changes and percentile bootstrap intervals
data/
    runs.csv                       912 evaluation runs, one row each (Section 5.1)
    calibration/calibration_runs.csv   24 WS calibration runs (same columns)
    calibration/ap_shares.json     AP target shares derived from the calibration runs
    packets/<experiment>/<run_id>.csv.gz   per-packet ledger of every run (936 files)
    traces/main_baseline_{RR,WS}_s3201.csv.gz   example time series (seed 3201)
    run_log.json                   software versions, worker count, wall time
results/
    tables/t1 ... t10 (.csv raw numbers, .md and .tex formatted)
    summary.md                     all tables in one file
    key_numbers.json               headline statistics in machine-readable form
    figures/fig1 ... fig5 (.pdf and .png)
    validation_report.txt          output of validate.py
    stress_sampling_check.csv/.md  output of check_stress_sampling.py
```

## 3. The benchmark in brief

| Element | Specification |
|---|---|
| Plant | 3 mechanically independent two-inertia axes, state [q_m, w_m, q_l, w_l]; J_m = 0.00324 kg m^2, J_l = 0.00567 kg m^2, k_s = 0.62046455 N m/rad (steel shaft, d_s = 2 mm, L_s = 0.20 m), c_s = 0.004, b_m = b_l = 0.0005 N m s/rad; f_f = 2.76092 Hz |
| Parameter mismatch | (J_m, J_l, k_s) of each axis perturbed uniformly by +/-12 % per trial; controller and estimator use nominal values |
| Discretisation | Exact zero-order hold via matrix exponentials, dt = 2 ms; trial length 16 s |
| References | r_i(t) = a_i sin(2 pi f_i t + phi_i)(1 - exp(-t/0.4)), a = (0.60, 0.35, 0.22) rad, f = (0.45, 0.27, 0.18) Hz, phi_i ~ U[-0.2, 0.2] |
| Disturbances | sin^2 load pulses, onsets 1 + 0.37 i + 2.4 n s (+/-0.15 s jitter), D ~ U[0.25, 0.55] s, amplitude +/-(0.025, 0.014, 0.009) N m; plus 0.002 z_j with z_j = 0.97 z_(j-1) + 0.03 w_j |
| Sensing | full-state packets y = x + v, v ~ N(0, diag(1e-8, 9e-6, 1e-8, 9e-6)) |
| Channel | one slot every h = 10 ms (1600 per trial); delay U{2..8} steps, loss 0.02; congestion when 2.2 < (t mod 3.2) < 2.7 s: delay U{15..30} steps, loss 0.15 |
| Estimator | state-reset predictor with input replay (main); in-order delayed Kalman filter (sensitivity) |
| Controller | continuous LQR, Q = diag(2, 0.003, 25, 0.015), R = 1, K = [3.312653, 0.158591, 1.883500, 0.462091]; u = sat_0.30{-K(x_hat - [r, 0, r, 0])} |
| Policies | RR, SA, RA, SO, WS, WS-B, SO-B, AP, TU (Table 2 of the article; `servo_bench/schedulers.py`) |
| Severity | rainflow on tau = 16 T / (pi d_s^3); F_m = sum n_c (a_c / tau_ref)^m, a_c = range / 2, tau_ref = 40 MPa, m = 3, 5, 7 |
| Statistics | paired change 100 (M_policy / M_RR - 1) within seed; mean over 24 seeds; 95 % percentile bootstrap, 20,000 resamples |

## 4. Experiment matrix

| Experiment (`experiment`) | Seeds | Configurations | Runs |
|---|---|---|---|
| Main benchmark (`main`) | 3201-3224 | 9 policies, state-reset predictor | 216 |
| Operating conditions (`operating`) | 4201-4224 | RR, RA, WS at 6 ms and 20 ms slots, doubled delay, 0 % and 25 % mismatch | 360 |
| Stress-weight sweep (`rho_sweep`) | 3201-3224 | WS at rho = 0.5, 1.5, 3, 12, 24, 48 (RR and rho = 6 from `main`) | 144 |
| Long horizon (`long_horizon`) | 5201-5224 | RR, WS over 64 s | 48 |
| Stiff transmission (`stiff`) | 6201-6224 | RR, RA, WS with d_s = 4 mm, k_s = 9.93 N m/rad, tau_ref = 5 MPa | 72 |
| Estimator robustness (`kalman`) | 3201-3224 | RR, WS, TU with the delayed Kalman filter | 72 |
| **Evaluation total** | | | **912** |
| AP calibration (`calibration`) | 2201-2224 | WS, used only to derive the AP target shares | 24 |

Condition labels in `runs.csv`: `baseline` (main), `slot6ms`, `slot20ms`, `delay2x`, `mismatch0`,
`mismatch25` (operating), `rho0.5` ... `rho48` (sweep), `t64s` (long horizon), `ds4mm` (stiff),
`delayed_kf` (Kalman).

## 5. Data dictionary

### 5.1 `data/runs.csv` and `data/calibration/calibration_runs.csv`

One row per run. Units are given in the column name where they apply; F-indices are dimensionless.

| Column | Meaning |
|---|---|
| experiment, condition, policy, seed | identify the run; runs with the same seed and condition share all exogenous draws |
| rho, t_run, slot_ms, delay_factor, mismatch, shaft_diameter, tau_ref, estimator | run configuration (s, ms, m, Pa) |
| run_id | `<experiment>__<condition>__<policy>__s<seed>`, also the packet-log file name |
| F3, F5, F7 | severity index summed over the three axes |
| F3_axis*i*, F5_axis*i*, F7_axis*i* | per-axis severity (axis 0, 1, 2) |
| rainflow_records | number of rainflow records (half and full cycles) over all axes |
| rmse_rad, rmse_rad_axis*i* | RMS tracking error r_i - q_l,i, pooled over axes and all 2 ms samples, and per axis |
| peak_tau_MPa, peak_tau_MPa_axis*i* | peak absolute surface shear stress |
| control_effort | sum over axes and steps of u^2 dt (N^2 m^2 s) |
| mean_age_ms, p99_age_ms | mean and 99th percentile of receiver age t - g_i, sampled every 2 ms, pooled over axes |
| max_age_ms_axis*i* | maximum receiver age of each axis |
| attempted, accepted, stale, lost, pending (and `_axis`*i*) | packet accounting; attempted = accepted + stale + lost + pending |
| aligned_corrections | accepted packets whose command change had the same sign as the true shaft twist |
| packets_file | relative path of the run's packet ledger |

### 5.2 `data/packets/<experiment>/<run_id>.csv.gz` (one row per slot)

| Column | Meaning |
|---|---|
| slot | slot index k |
| axis | axis the scheduler selected |
| gen_step | internal step at which the sample was taken (k h / dt) |
| arr_step | arrival step (-1 if lost; >= trial length if still pending at the end) |
| delay_steps | channel delay in 2 ms steps (-1 if lost) |
| congested | 1 if the slot fell in a congestion interval |
| status | accepted, stale, lost or pending |
| torque_true_Nm | true elastic torque k_s,i (q_m - q_l) at the arrival step (accepted and stale) |
| torque_est_before_Nm, torque_est_after_Nm | controller's torque estimate just before and after the update (accepted) |
| du_cmd_Nm | change in the saturated motor command caused by the update (accepted) |
| aligned | 1 if du_cmd and the true twist have the same sign, 0 if not, -1 if not accepted |

### 5.3 Other data files

* `data/calibration/ap_shares.json`: calibration seeds, raw mean request shares of WS, the shares rounded to five decimals that AP uses, and the resulting AP allocation per 16 s run.
* `data/traces/main_baseline_{RR,WS}_s3201.csv.gz`: time (s), reference and load angle (rad), surface shear stress (MPa) and motor torque (N m) for each axis at 2 ms resolution.
* `data/run_log.json`: Python and library versions, worker count and wall-clock time of the archive build.

## 6. Randomness and pairing

Each seed initialises one `numpy.random.SeedSequence`, which is split into six independent streams:
parameter perturbations, reference phases, load pulses, colored-disturbance noise, sensor noise and
channel outcomes. All draws are made before the closed loop starts and are indexed only by time step
or slot (sensor noise is drawn for every slot and every axis, whether or not that axis is scheduled).
Scheduling decisions therefore cannot change the disturbance, noise or channel realisation, and every
policy evaluated on a seed sees exactly the same environment. Mismatch levels share one set of unit
draws, and the doubled-delay condition doubles the same delay draws.

## 7. Implementation choices where the article leaves details open

1. **Step ordering** within step j: deliver due packets (oldest generation first), then let the scheduler choose at slot steps, then compute u_j from the updated estimates, then advance the plants and the estimates over [t_j, t_(j+1)).
2. **Staleness**: a packet is accepted only if its sample is strictly newer than the newest accepted sample of its axis.
3. **Initial conditions**: plants at rest (x = 0), estimates x_hat = 0, g_i = 0, s_i = -dt; delayed-KF posterior mean 0 with covariance R_v.
4. **Congestion** is evaluated at each slot's transmission time with exact integer step arithmetic (boundaries excluded).
5. **Pulses** with nominal onset before the end of the trial are included; a pulse that runs past the end is truncated. The colored process starts from z_(-1) = 0.
6. **Cycle amplitude** is half the rainflow range (a_c = range / 2).
7. **Tracking RMSE** compares the load angle q_l with r_i, pooled over the three axes and every 2 ms sample.
8. **Ties** in every argmax are broken toward the lowest axis index.
9. **TU** propagates the measurement covariance R_v with the nominal open-loop model and no process noise: V(n) = c^T A_d^n R_v (A_d^n)^T c. V(n) rises monotonically over the request ages that occur, so TU selects the same axis as RR at every slot in this benchmark (validate.py, D11).
10. **Stiff shaft**: the LQR is re-synthesised with the same Q and R for the 4 mm nominal plant, giving K = [4.454580, 0.176476, 0.741572, 0.189511]; predictor and filter also use the 4 mm nominal model.
11. **AP shares** are the mean per-axis request shares of WS over calibration seeds 2201-2224, rounded to five decimals.
12. **Packet diagnostics**: the high-torsion threshold is the 80th percentile of the true torque magnitude at accepted axis-0 arrivals under RR, pooled over the 24 main seeds. A correction reinforces the twist when the change in saturated command caused by the update has the same sign as the true shaft twist q_m - q_l.
13. **Bootstrap** intervals use 20,000 resamples of the 24 paired seeds with generator seed 20260611; every interval uses the same resample indices, so results do not depend on the order of computation.

## 8. Verification

`validate.py` runs 114 checks in nine groups: hand-countable rainflow benchmarks (including the
ASTM E1049 worked example), scale, offset and sign invariance, agreement with an independent
re-implementation of the rainflow procedure, exact discretisation against an RK4 solver,
matrix-exponential semigroup identities, Van Loan covariances against direct quadrature and positive
definiteness, the published LQR gain and closed-loop eigenvalues, Kalman updates against batch
Gaussian conditioning, channel and scheduler rules, bit-identical reproducibility (in-process, in a
fresh process and against archived rows), packet conservation and per-axis summation over all 936
runs, the algebraic equivalence of SA and RR, and a re-run of the RR baseline for all 24 main seeds
against the archive. All 114 checks pass on the archive (`results/validation_report.txt`).

`check_stress_sampling.py` reconstructs the shaft stress at 0.5 ms between control updates without
changing decisions or inputs. The largest change in F5 is 0.072 % on the baseline shaft and 0.387 % on
the stiff shaft (12 runs each), so the 2 ms grid resolves the rainflow cycles adequately.

## 9. Headline results regenerated by `analyze.py`

Main benchmark, 24 paired seeds, change relative to RR (full tables in `results/summary.md`):

| Comparison | dF5, mean [95 % bootstrap interval] | Seeds up / down / tie |
|---|---|---|
| WS vs RR | +2.69 % [+0.52 %, +4.88 %] | 18 / 6 / 0 |
| RA vs RR | +4.58 % [+1.65 %, +7.85 %] | 17 / 7 / 0 |
| SO-B vs RR | -1.22 % [-3.25 %, +0.84 %] | 9 / 15 / 0 |
| AP vs RR | -0.50 % [-2.74 %, +1.70 %] | 10 / 14 / 0 |
| WS vs AP | +3.51 % [+0.59 %, +6.93 %] | 16 / 8 / 0 |

Tracking RMSE changed by -0.0297 % under WS. WS raised F7 more than F3 (+4.78 % vs +1.08 %).
SO-B left the aggregate unchanged but raised F5 on axes 1 and 2 by about 37 %; unguarded SO starved
axis 2 for up to 16.0 s. On axis 0, WS increased high-torsion arrivals from 102.0 to 113.0 per run
and reinforcing high-torsion corrections by 9.71 [7.79, 11.71] per run. The WS penalty remained
positive under every operating condition tested, including 6 ms slots (+3.29 % [+2.12 %, +4.50 %]),
the delayed Kalman filter (+2.62 % [+0.45 %, +4.82 %]) and the stiff shaft (+28.2 % [+6.3 %, +52.3 %],
14 of 24 seeds up). No stress-weighting gain between 0.5 and 48 reduced F5 significantly.

## 10. Software environment and reproducibility

The archive was built with Python 3.13.18, NumPy 2.5.3, SciPy 1.18.1, pandas 3.0.5 and
Matplotlib 3.11.2 on Linux x86-64 (`data/run_log.json`). With these versions, `run_experiments.py`
reproduces `data/runs.csv` exactly. Other NumPy, SciPy or BLAS versions can differ in the last
floating-point digit of a matrix exponential; such differences can occasionally flip an individual
scheduling decision and so perturb single runs slightly, while the aggregate statistics agree closely.
The random streams themselves (PCG64 via `SeedSequence`) are stable across NumPy versions.

## 11. Licence and citation

Code and data are released under the Creative Commons Attribution 4.0 International licence.
Please cite the dataset as:

> Al Junayed, A., Ibn Muzahid, L. (2026). Replication package: Stress-weighted telemetry and
> stress-cycle severity in flexible servo axes. Mendeley Data, V1. https://doi.org/10.17632/hbcy5m7rrk.1
