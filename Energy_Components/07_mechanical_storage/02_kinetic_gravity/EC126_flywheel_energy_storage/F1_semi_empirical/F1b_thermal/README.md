# EC126 — Flywheel Energy Storage — F1b Thermal

## Overview
Extends F1a kinetic model with **speed-dependent windage and bearing losses** and **temperature effects** on air density.

## Physics
- Windage losses: `P_windage = k_w * omega^3` (cubic with angular velocity)
- Bearing losses: `P_bearing = k_b * omega` (linear for magnetic bearings)
- Temperature effect: `k_w(T) = k_w_ref * rho_air(T)/rho_air_ref` (ideal gas scaling)
- Self-discharge rate: `(P_windage + P_bearing) / E_stored` [1/h]
- SOC proportional to omega^2: `SOC = (omega^2 - omega_min^2) / (omega_max^2 - omega_min^2)`

## Inputs
| Parameter | Unit | Default | Range |
|-----------|------|---------|-------|
| soc | - | - | [0, 1] |
| power_command_kw | kW | 0 | [-100, 100] |
| ambient_temperature | degC | 25 | [-20, 60] |

## Outputs
| Parameter | Unit |
|-----------|------|
| power_actual_kw | kW |
| losses_kw | kW |
| self_discharge_rate_per_hour | 1/h |
| efficiency | - |
| speed_rpm | rpm |

## Default Parameters
- J=100 kg*m2, omega_max=30000 rpm, omega_min=15000 rpm
- E_max=100 kWh, P_rated=100 kW
- k_windage=1e-10 W/(rad/s)^3, k_bearing=0.001 W/(rad/s)

## References
- Arani et al. (2017). Energies, 10, 1361.
- Beacon Power (2011). Flywheel Technical Report.
- Genta, G. (2005). Kinetic Energy Storage. Butterworth-Heinemann.
