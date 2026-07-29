# EC153 — Binary Cycle Geothermal Plant — F1a Exergy Efficiency Model

## Overview
Exergy-based efficiency model for organic Rankine cycle (ORC) binary geothermal plants.
Plant efficiency = utilization efficiency × Carnot efficiency. Power output scales with
brine flow rate and available temperature differential.

## Model Equations
```
eta_Carnot  = 1 - T_rejection / T_geothermal   [temperatures in Kelvin]
eta_plant   = eta_utilization * eta_Carnot       [eta_util = 0.45]
T_reinject  = T_rejection + 10                  [degC, minimum reinjection]
Q_heat      = m_dot * cp * (T_geo - T_reinject) [kW]
P           = Q_heat * eta_plant                 [kW]
```

## Inputs
| Name | Unit | Range | Description |
|------|------|-------|-------------|
| T_geothermal | degC | [80, 200] | Geothermal brine inlet temperature |
| T_rejection | degC | [10, 40] | Condenser/cooling rejection temperature |
| flow_rate_kgs | kg/s | [10, 100] | Geothermal brine mass flow rate |

## Outputs
| Name | Unit | Description |
|------|------|-------------|
| power_kw | kW | Net electrical power output |
| efficiency | - | Overall plant thermal efficiency |
| heat_input_kw | kW | Thermal energy extracted from brine |
| T_reinjection_c | degC | Brine reinjection temperature |

## Design Point (T_geo=150°C, T_rej=25°C, 50 kg/s)
| Parameter | Value |
|-----------|-------|
| Power | ~2,700 kW |
| Efficiency | ~10.7% |
| Heat Input | ~25,200 kW |
| T_reinjection | 35°C |

## Sources
1. DiPippo, R. (2015). *Geothermal Power Plants: Principles, Applications, Case Studies and Environmental Impact*, 4th ed. Butterworth-Heinemann.

## Limitations
- Fixed utilization efficiency (0.45); actual value depends on working fluid, HX design, pump losses
- No partial-load or start-up transients
- Brine treated as liquid water (cp = 4186 J/kgK); salinity and dissolved gases ignored
- Flash and direct-steam plant types not modeled
