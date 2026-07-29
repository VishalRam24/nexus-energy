# EC109 — Simple Cycle Gas Turbine — F1a Efficiency Curve

## Overview
Semi-empirical part-load and ambient-temperature efficiency model for a simple-cycle open-Brayton gas turbine, parameterized on a GE LM6000-class machine (~43 MW).

## Model Equations
```
eta(PLR, T_amb) = eta_rated * f_PLR(PLR) * f_amb(T_amb)
f_PLR = b0 + b1*PLR + b2*PLR^2     [peaked parabolic, coefficients from Walsh & Fletcher]
f_amb = 1 - 0.007 * (T_amb - 15)   [ISO derating: 0.7% per °C above ISO]
P_out = P_rated * PLR               [MW]
fuel_rate = (P_out / eta) / LHV     [kg/s, LHV = 50 MJ/kg natural gas]
heat_rate  = 3600 / eta             [kJ/kWh]
```

## Inputs
| Name | Unit | Range | Description |
|------|------|-------|-------------|
| part_load_ratio | — | [0.3, 1.0] | Fraction of rated capacity |
| ambient_temp_c | degC | [-20, 50] | Inlet ambient temperature |

## Outputs
| Name | Unit | Description |
|------|------|-------------|
| power_mw | MW | Net electrical output |
| efficiency | — | Net LHV electrical efficiency |
| fuel_rate_kgs | kg/s | Natural gas consumption |
| heat_rate_kjkwh | kJ/kWh | Heat rate (lower is better) |

## Design Parameters
| Parameter | Value | Unit |
|-----------|-------|------|
| P_rated | 43 | MW |
| eta_rated | 0.41 | LHV |
| T_amb_ref | 15 | degC (ISO) |
| b0, b1, b2 | 0.1, 1.1, -0.2 | — |
| f_amb coeff | 0.007 | 1/°C |

## Sources
1. Walsh & Fletcher (2004), "Gas Turbine Performance", 2nd ed., Blackwell Science.
2. GE LM6000 product family data sheet (indicative, publicly available).

## Limitations
- Steady-state only; no start-up/shutdown transient
- Fuel fixed to natural gas LHV = 50 MJ/kg
- No inlet fogging, evaporative cooling, or inlet heating modeled
- Ambient derating is linear — non-linear at extreme temperatures
