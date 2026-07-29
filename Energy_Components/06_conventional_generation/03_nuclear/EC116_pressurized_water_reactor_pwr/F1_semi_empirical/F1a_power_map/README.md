# EC116 — Pressurized Water Reactor (PWR) — F1a Steady-State Power Map

## Overview
Steady-state power balance model for a 3000 MWth / 1000 MWe 4-loop PWR. Maps part-load ratio to electrical output via Rankine cycle efficiency, with coolant temperature calculation.

## Model Equations
```
P_thermal    = P_thermal_rated * PLR                      [MW_th]
f_PLR        = 1.0                    for PLR >= 0.5
             = PLR / 0.5             for PLR < 0.5
P_electric   = P_thermal * eta_cycle * eta_gen * f_PLR   [MW_e]
eta_net      = P_electric / P_thermal                     [-]
dT           = P_thermal * 1e6 / (m_dot * cp * 1000)     [degC]
T_outlet     = T_inlet + dT                               [degC]
```

## Inputs
| Name | Unit | Range | Description |
|------|------|-------|-------------|
| part_load_ratio | - | [0.5, 1.0] | Fraction of rated thermal power |
| coolant_flow_kgs | kg/s | [9000, 18000] | Primary coolant mass flow (optional) |

## Outputs
| Name | Unit | Description |
|------|------|-------------|
| electric_power_mw | MW_e | Net electrical output |
| thermal_power_mw | MW_th | Reactor thermal power |
| efficiency | - | Net thermal efficiency |
| coolant_outlet_temp_c | degC | Hot-leg coolant temperature |

## Parameters (Generic 4-loop Westinghouse PWR)
| Parameter | Value | Unit | Source |
|-----------|-------|------|--------|
| P_thermal_rated | 3000 | MW_th | Todreas & Kazimi (2012) |
| P_electric_rated | 1000 | MW_e | Todreas & Kazimi (2012) |
| eta_cycle | 0.33 | - | Rankine steam cycle |
| eta_gen | 0.99 | - | Turbogenerator efficiency |
| T_inlet | 292 | degC | Cold-leg temperature |
| T_outlet_rated | 326 | degC | Hot-leg at full power |
| P_pressure | 155 | bar | Primary circuit |
| PLR_min | 0.5 | - | Minimum stable load |
| m_dot_nominal | 18000 | kg/s | Total 4-loop flow |

## Sources
1. Todreas, N.E. & Kazimi, M.S. (2012). Nuclear Systems: Volume I, 2nd Edition. CRC Press.

## Physics Checks
- eta ~ 0.33 (Rankine cycle limit for PWR steam conditions)
- P_electric = P_thermal * eta at all PLR values
- PLR_min = 0.5 enforced (nuclear minimum stable power)
- T_outlet increases with PLR (more heat generated per unit coolant flow)

## Limitations
- Steady-state only — no point kinetics, xenon transients, or load-following dynamics
- No fuel burnup or reactivity feedback modeling
- Single-node coolant temperature (no axial distribution)
- No primary/secondary loop thermal resistance modeling
