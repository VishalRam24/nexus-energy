# EC097 — Rankine Cycle (Steam Turbine) — F1a Efficiency Curve

## Overview
Simple part-load efficiency model for subcritical Rankine cycle steam turbines. Predicts cycle efficiency, power output, heat input, and steam flow as a function of part-load ratio with Carnot limit enforcement.

## Inputs
| Name | Unit | Range | Description |
|------|------|-------|-------------|
| PLR | - | [0, 1] | Part-load ratio (P/P_rated) |
| T_steam | degC | [400, 620] | Steam temperature (optional) |
| T_condenser | degC | [20, 50] | Condenser temperature (optional) |

## Outputs
| Name | Unit | Description |
|------|------|-------------|
| efficiency | - | Cycle thermal efficiency |
| power_output | W | Electrical power output |
| heat_input | W | Thermal heat input |
| steam_flow | kg/s | Approximate steam mass flow rate |
| carnot_efficiency | - | Carnot upper bound |

## Equations
```
eta = eta_rated * (1 - a*(1-PLR)^2)
eta_carnot = 1 - T_cold/T_hot
eta = min(eta, eta_carnot)
Q_in = P_out / eta
m_dot = Q_in / h_drop
```

## Parameters
Typical subcritical unit: P_rated=100 MW, eta_rated=0.38, T_steam=540 degC, P_steam=170 bar.

## Sources
1. Cotton (1998). Evaluating and Improving Steam Turbine Performance.
2. Lior (2002). "Power from steam." Energy Conversion and Management.

## Limitations
- No off-design steam condition modeling (fixed enthalpy drop estimate)
- No multi-stage turbine detail
- No startup/shutdown transients
- Part-load curve is a simple quadratic fit
