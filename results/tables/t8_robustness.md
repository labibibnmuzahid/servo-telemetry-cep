### Paired change in F5 relative to RR across operating conditions

| Operating condition | WS dF5 [95% CI] | WS incr. / decr. / tie | WS sign-test p | WS median | RA dF5 [95% CI] | Seeds |
|---|---|---|---|---|---|---|
| Main benchmark (10 ms slots, 12% mismatch) | +3.79% [+2.59%, +5.13%] | 73 / 23 / 0 | 3.1e-07 | +2.80% | +4.36% [+2.84%, +5.96%] | 96 |
| Fast telemetry (6 ms slots) | +2.16% [+1.54%, +2.77%] | 74 / 22 / 0 | 9.4e-08 | +2.07% | +4.21% [+2.95%, +5.52%] | 96 |
| Slow telemetry (20 ms slots) | +5.10% [+2.51%, +7.41%] | 76 / 20 / 0 | 7.3e-09 | +5.62% | +0.14% [-2.35%, +2.43%] | 96 |
| Doubled channel delay | +4.74% [+3.22%, +6.30%] | 72 / 24 / 0 | 9.7e-07 | +4.78% | +11.78% [+9.06%, +14.56%] | 96 |
| 25% parameter mismatch | +4.08% [+3.04%, +5.17%] | 80 / 16 / 0 | 2.1e-11 | +3.83% | +6.93% [+4.31%, +9.92%] | 96 |
| Nominal plant (no mismatch) | +4.24% [+3.15%, +5.22%] | 83 / 13 / 0 | 1.2e-13 | +4.39% | +4.05% [+2.51%, +5.57%] | 96 |
| Extended 64 s horizon | +5.19% [+4.34%, +6.09%] | 89 / 7 / 0 | 3.3e-19 | +4.98% | - | 96 |
| Delayed Kalman filter | +3.78% [+2.57%, +5.11%] | 72 / 24 / 0 | 9.7e-07 | +2.80% | - | 96 |
| Stiff transmission (d_s = 4 mm) | +20.59% [+5.12%, +40.06%] | 47 / 49 / 0 | 0.92 | -1.78% | +328.33% [+257.22%, +407.39%] | 96 |

Each condition is compared with RR on the same seeds and in the same condition. Under the delayed Kalman filter, TU vs RR is reported in key_numbers.json.
