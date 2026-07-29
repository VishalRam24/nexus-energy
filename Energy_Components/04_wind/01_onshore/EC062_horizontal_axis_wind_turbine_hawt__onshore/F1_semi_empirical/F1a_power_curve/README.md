# EC062 — HAWT Onshore Wind Turbine — F1a Power Curve Model

## Overview
Power curve P(v) model with air density correction for onshore horizontal axis wind turbines. Based on manufacturer power curve interpolation per IEC 61400-12-1.

## Inputs
| Name | Unit | Range | Description |
|------|------|-------|-------------|
| wind_speed | m/s | [0, 30] | Hub-height wind speed |
| air_density | kg/m3 | [0.9, 1.4] | Air density (default 1.225) |

## Outputs
| Name | Unit | Description |
|------|------|-------------|
| power_kw | kW | Electrical power output |
| capacity_factor | - | P / P_rated |
| power_coefficient | - | Cp (aerodynamic efficiency) |

## Sources
1. IEC 61400-12-1 standard
2. Manwell et al. (2009). "Wind Energy Explained," 2nd ed.
3. windpowerlib v0.2.2 (MIT), Vestas V90-2.0MW

## Limitations
- No turbulence effects, no wake modeling
- No yaw misalignment, no pitch control dynamics
- Linear interpolation of power curve
