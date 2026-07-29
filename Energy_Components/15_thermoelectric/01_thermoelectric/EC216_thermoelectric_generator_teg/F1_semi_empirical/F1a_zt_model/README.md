# EC216 — Thermoelectric Generator (TEG) — F1a ZT Model

## Overview
ZT-based efficiency and power model for a Bi2Te3 TEG module (40x40mm, 127 thermoelectric couples).
Uses the standard thermoelectric efficiency formula derived from ZT figure of merit with Carnot
reduction, and matched-load power from Seebeck coefficient and internal resistance.

## Model Equations
```
eta_Carnot = 1 - T_cold / T_hot                               [K basis]
eta = eta_Carnot * (sqrt(1+ZT) - 1) / (sqrt(1+ZT) + T_c/T_h)
P_max = (alpha*N)^2 * dT^2 / (4 * R_int)                     [matched load]
Q_hot = P_max / eta                                           [W]
V_load = 0.5 * alpha * N * dT                                 [V at matched load]
```

## Inputs
| Name | Unit | Range | Description |
|------|------|-------|-------------|
| T_hot | degC | [50, 300] | Hot side temperature |
| T_cold | degC | [0, 50] | Cold side (heat sink) temperature |

## Outputs
| Name | Unit | Description |
|------|------|-------------|
| efficiency | - | Thermoelectric conversion efficiency |
| power_w | W | Electrical power output (matched load) |
| heat_input_w | W | Heat input at hot side |
| voltage_v | V | Terminal voltage at matched load |

## Parameters
| Parameter | Value | Unit | Source |
|-----------|-------|------|--------|
| ZT | 1.0 | - | Typical Bi2Te3 (Snyder & Toberer 2008) |
| N_couples | 127 | - | Standard 40x40mm module |
| module_area | 0.0016 | m2 | 40mm x 40mm |
| alpha_seebeck | 200 | uV/K | Bi2Te3 per couple |
| R_internal | 2.0 | ohm | Per module at matched load |

## Sources
1. Rowe, D.M. (ed.) (2006). *Thermoelectrics Handbook: Macro to Nano*. CRC Press.
2. Snyder, G.J. & Toberer, E.S. (2008). Complex thermoelectric materials.
   *Nature Materials*, 7, 105-114.

## Limitations
- ZT assumed constant (temperature-independent) — valid for narrow temperature ranges
- Matched load condition only (no load line analysis)
- Single-module model (no thermal/electrical stacking)
- No contact resistance or thermal interface material resistance
