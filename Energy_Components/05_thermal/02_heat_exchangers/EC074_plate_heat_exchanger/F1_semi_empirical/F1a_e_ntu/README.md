# EC074 — Plate Heat Exchanger — F1a Effectiveness-NTU Model

## Overview
Counter-flow effectiveness-NTU (ε-NTU) model for a brazed plate heat exchanger operating with water on both sides.

## Model Equations
```
NTU = U * A / C_min
C_r = C_min / C_max
epsilon = (1 - exp(-NTU*(1-C_r))) / (1 - C_r*exp(-NTU*(1-C_r)))   [C_r < 1]
epsilon = NTU / (1 + NTU)                                           [C_r = 1]
Q = epsilon * C_min * (T_h_in - T_c_in)
T_h_out = T_h_in - Q / C_h
T_c_out = T_c_in + Q / C_c
```

## Inputs
| Name | Unit | Range | Description |
|------|------|-------|-------------|
| T_h_in | degC | [30, 120] | Hot-side inlet temperature |
| T_c_in | degC | [5, 60] | Cold-side inlet temperature |
| m_dot_hot | kg/s | [0.05, 5] | Hot-side mass flow rate |
| m_dot_cold | kg/s | [0.05, 5] | Cold-side mass flow rate |

## Outputs
| Name | Unit | Description |
|------|------|-------------|
| Q_kw | kW | Heat transfer rate |
| T_h_out | degC | Hot-side outlet temperature |
| T_c_out | degC | Cold-side outlet temperature |
| effectiveness | — | ε = Q / Q_max |
| ntu | — | Number of Transfer Units |

## Design Parameters
| Parameter | Value | Unit |
|-----------|-------|------|
| U | 3000 | W/m²K |
| A | 2.0 | m² |
| cp (both fluids) | 4186 | J/kgK |

## Sources
1. Incropera & DeWitt (2006), Fundamentals of Heat and Mass Transfer, 6th ed., ch. 11.
2. Shah & Sekulic (2003), Fundamentals of Heat Exchanger Design, Wiley.

## Limitations
- Constant U assumption (no fouling model — see F1b for fouling correction)
- Steady-state only (no thermal inertia)
- Both fluids treated as pure water; no two-phase flow
- Counter-flow configuration only
