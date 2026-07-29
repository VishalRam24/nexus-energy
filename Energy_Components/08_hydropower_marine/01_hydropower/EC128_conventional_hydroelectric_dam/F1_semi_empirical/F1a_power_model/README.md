# EC128 — Conventional Hydroelectric Dam — F1a Power Model

## Overview
Semi-empirical power model for a Francis turbine hydroelectric plant.
Power output is computed from the hydraulic power equation with a flow-dependent turbine efficiency curve.

## Model Equations
```
P = eta_overall * rho * g * Q * H / 1000   [kW]
eta_overall = eta_turbine(q) * eta_generator
eta_turbine(q) = eta_peak * (1 - k * (q - 1)^2),   q = Q / Q_design
```
Valid for q in [0.3, 1.1]; zero output outside this range (turbine off or runaway).

## Inputs
| Name | Unit | Range | Description |
|------|------|-------|-------------|
| flow_rate_m3s | m3/s | [0, 33] | Penstock flow rate |
| head_m | m | [50, 150] | Net hydraulic head |

## Outputs
| Name | Unit | Description |
|------|------|-------------|
| power_kw | kW | Electrical output power |
| turbine_efficiency | — | Hydraulic-to-shaft efficiency |
| overall_efficiency | — | Shaft-to-grid efficiency |
| capacity_factor | — | Power / P_rated |

## Design Parameters
| Parameter | Value | Unit |
|-----------|-------|------|
| H_design | 100 | m |
| Q_design | 30 | m³/s |
| P_rated | 27,000 | kW |
| eta_peak | 0.93 | — |
| eta_generator | 0.98 | — |
| k (efficiency drop) | 0.3 | — |

## Sources
1. Dixon & Hall (2014), "Fluid Mechanics and Thermodynamics of Turbomachinery", 7th ed., Butterworth-Heinemann.
2. IEC 60041 — Field acceptance tests to determine the hydraulic performance of hydraulic turbines.

## Limitations
- No hydraulic transients (water hammer), cavitation, or sediment effects
- Single turbine unit; no parallel unit dispatch
- Head assumed constant (no reservoir depletion model)
- Valid only for Francis turbine family; different k for Kaplan or Pelton
