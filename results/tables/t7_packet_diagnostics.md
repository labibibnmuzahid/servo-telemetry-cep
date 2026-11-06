### Packet diagnostics on axis 0 (main benchmark, 96 seeds)

| Quantity | RR | WS |
|---|---|---|
| High-torsion threshold: 80th percentile of abs(T) at accepted axis-0 arrivals under RR | 26.243 mN m |  |
| Accepted axis-0 packets per run | 509.57 | 519.01 |
| High-torsion axis-0 arrivals per run | 101.92 | 111.68 |
| ... of which the command change reinforced the twist | 80.32 | 89.42 |
| Paired difference WS - RR in reinforcing high-torsion arrivals |  | +9.09 [+8.06, +10.16] |
| Share of high-torsion arrivals that reinforced the twist | 78.8% | 80.1% |
| Mean abs torque-estimate error just before an axis-0 update | 1.009 mN m | 1.048 mN m |

A correction reinforces the twist when the change in saturated motor command caused by the update has the same sign as the true shaft twist q_m - q_l at the arrival step.
