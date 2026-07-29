# EC168 — MPPT Controller — F1a Tracking Efficiency Model

## Overview
Exponential saturation model for MPPT tracking efficiency as a function of irradiance.
eta_mppt = eta_max * (1 - exp(-k * G / G_ref)), where k controls the low-irradiance roll-off.

## Inputs
| Name | Unit | Range | Description |
|------|------|-------|-------------|
| irradiance | W/m2 | [0, 1200] | Solar irradiance at module plane |
| p_mpp_input | W | [0, 12000] | Available MPP power from PV array |

## Outputs
| Name | Unit | Description |
|------|------|-------------|
| p_output | W | MPPT controller output power |
| tracking_efficiency | - | Tracking efficiency (eta_mppt) |
| power_loss | W | Power lost due to imperfect tracking |

## Parameters
- eta_max: 0.99 (maximum tracking efficiency at high irradiance)
- k: 5.0 (low-irradiance roll-off steepness)
- G_ref: 1000 W/m2 (STC reference irradiance)
- P_rated: 10 kW

## Behavior
- G > 200 W/m2: eta ~ 0.98–0.99
- G < 50 W/m2:  eta drops to ~0.90–0.92
- G = 0: eta = 0 (no power)

## Sources
1. Hohm, D.P. & Ropp, M.E. (2003). Comparative study of maximum power point tracking algorithms.
   Progress in Photovoltaics, 11, 47-62.
2. De Brito, M.A.G. et al. (2013). IEEE Trans. Ind. Electron., 60(3), 1156-1167.

## Limitations
- Static model (no transient response or dynamic tracking lag)
- No temperature dependence of tracking performance
- No algorithm-specific behavior (P&O, INC, etc. all map to same eta curve)
