# EC128 — Conventional Hydroelectric Dam — F1b Head-Flow

## Overview
Extends F1a power model with **2D turbine efficiency hill chart** (flow and head dependence), **multi-turbine type support** (Francis, Kaplan, Pelton), and **environmental flow constraint**.

## Physics
- Hill chart: `eta(q,h) = eta_peak * (1 - k_q*(q-1)^2) * (1 - k_h*(h-1)^2)`
  - q = Q/Q_rated (flow ratio), h = H/H_rated (head ratio)
- Three turbine types with different operating envelopes:
  - Francis: 40-600m head, eta_peak=0.93
  - Kaplan: 2-40m head, eta_peak=0.91, flatter curve (adjustable blades)
  - Pelton: 300-1800m head, eta_peak=0.91
- Environmental flow: minimum 10% of Q_rated must be released

## Inputs
| Parameter | Unit | Range |
|-----------|------|-------|
| flow_rate_m3s | m3/s | [0, 200] |
| head_m | m | [5, 1800] |
| turbine_type | - | francis/kaplan/pelton |

## Outputs
| Parameter | Unit |
|-----------|------|
| power_kw | kW |
| efficiency | - |
| specific_speed | - |
| flow_ratio | - |

## Default Parameters (Francis)
- Q_rated=50 m3/s, H_rated=100m, P_rated=40 MW
- eta_peak=0.93, eta_generator=0.98
- Environmental flow=10% of Q_rated

## References
- Dixon, S.L. & Hall, C.A. (2014). Fluid Mechanics and Thermodynamics of Turbomachinery, 7th ed.
- IEC 60041:1991 — Field acceptance tests for hydraulic turbines.
