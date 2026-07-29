# EC065 — Offshore Fixed-Bottom Wind Turbine — F1a Power Curve Model

## Overview
Interpolated manufacturer power curve with linear air density correction.
Turbine: Siemens SWT-3.6-120 (3.6 MW, 120 m rotor, 90 m hub).
P(v, rho) = P_curve(v) * (rho / rho_ref)

## Inputs
| Name | Unit | Range | Description |
|------|------|-------|-------------|
| wind_speed | m/s | [0, 30] | Hub-height (90 m) wind speed |
| air_density | kg/m3 | [0.9, 1.4] | Air density (default 1.225 = STC) |

## Outputs
| Name | Unit | Description |
|------|------|-------------|
| power_kw | kW | Electrical output power |
| capacity_factor | - | Power / rated power (3600 kW) |
| power_coefficient | - | Cp = P / (0.5 * rho * A * v^3) |

## Turbine Parameters
- P_rated: 3600 kW
- Rotor diameter: 120 m, hub height: 90 m
- Cut-in: 3.5 m/s, Rated: ~14 m/s (full rated at 14 m/s), Cut-out: 25 m/s
- Rotor area: 11310 m2

## Power Curve Data (Siemens SWT-3.6-120)
Wind speeds 0–25 m/s at 1 m/s intervals with 0 kW at cut-in (3.5 m/s), ramping to 3600 kW by 14 m/s.

## Sources
1. IEC 61400-12-1:2017 — Power performance measurements.
2. windpowerlib (MIT): https://github.com/wind-python/windpowerlib
3. Siemens SWT-3.6-120 product datasheet.

## Limitations
- No turbulence intensity correction
- No wake effects (single turbine model)
- Linear density correction (ignores compressibility at high altitude)
- No yaw misalignment or blade pitch dynamics
- Offshore wind shear alpha=0.11 (lower than onshore due to lower surface roughness)
