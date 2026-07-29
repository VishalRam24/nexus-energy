# EC122 — Pumped Hydro Storage — F1b Head Variation

## Overview
Extends F1a round-trip model with **variable head as a function of SOC** and **penstock friction losses** via Darcy-Weisbach.

## Physics
- Variable head: `H(SOC) = H_min + SOC * (H_max - H_min)`
- Penstock friction: `h_f = f * L * v^2 / (2 * D * g)` where `v = Q / A_penstock`
- Generation: `P_gen = eta_t * eta_g * rho * g * Q * (H - h_f) / 1000` [kW]
- Pumping: `P_pump = rho * g * Q * (H + h_f) / (eta_p * eta_m * 1000)` [kW]
- Round-trip efficiency accounts for friction in both directions

## Inputs
| Parameter | Unit | Range |
|-----------|------|-------|
| SOC | - | [0, 1] |
| flow_rate_m3s | m3/s | [0, 100] |
| mode | - | "charge" / "discharge" |

## Outputs
| Parameter | Unit |
|-----------|------|
| power_kw | kW |
| effective_head_m | m |
| friction_loss_m | m |
| efficiency | - |
| round_trip_efficiency | - |

## Default Parameters
- H_max=500m, H_min=480m, penstock L=1000m D=3m
- eta_turbine=0.90, eta_pump=0.88, eta_generator=0.97, eta_motor=0.97
- Darcy friction factor f=0.015

## References
- Rehman et al. (2015). RSER, 44, 586-598.
- Mosonyi, E. (1991). Water Power Development. Akademiai Kiado.
- Munson et al. (2013). Fluid Mechanics, 7th ed. Wiley.
