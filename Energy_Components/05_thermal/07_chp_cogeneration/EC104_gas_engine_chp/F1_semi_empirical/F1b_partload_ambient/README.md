# EC104 -- Gas Engine CHP -- F1b Part-Load + Ambient Temperature

## Overview
Extends F1a by adding quadratic electrical efficiency curve, ambient temperature derating, and a thermal efficiency model that captures the increasing heat-to-power ratio at part load.

## Model Equations
```
Electrical efficiency:
    eta_el(PLR) = eta_el_rated * (a + b*PLR + c*PLR^2) * f_temp(T_amb)

Thermal efficiency:
    eta_th(PLR) = eta_th_rated * (th_a + th_b*PLR)
    (Increases proportionally at part load -- more jacket heat recovered)

Total efficiency:
    eta_total = eta_el + eta_th

Heat-to-power ratio:
    HPR = eta_th / eta_el

Temperature derating (above 25 degC):
    f_temp = 1 - 0.003 * max(0, T_amb - 25)
```

## Inputs
| Name | Unit | Range | Default |
|------|------|-------|---------|
| PLR | - | [0.5, 1.0] | - |
| T_ambient | degC | [-20, 50] | 25.0 |

## Outputs
| Name | Unit | Description |
|------|------|-------------|
| efficiency_electrical | - | Electrical efficiency |
| efficiency_thermal | - | Thermal efficiency |
| efficiency_total | - | Total first-law efficiency |
| power_electrical_kw | kW_e | Electrical output |
| heat_recovery_kw | kW_th | Thermal heat recovery |
| fuel_input_kw | kW_fuel | Fuel input power |
| heat_to_power_ratio | - | Q_th / P_el |

## Key Physics
- Electrical efficiency drops at part load (off-design engine operation)
- Thermal efficiency is relatively higher at part load (more proportional jacket heat)
- Total CHP efficiency remains high (75-90%) across operating range
- HPR increases at part load (more heat per unit of electricity)
- Ambient temperature above 25 degC reduces charge air density and power

## Sources
1. US EPA CHP Catalog (2017). CHP Technology Fact Sheets.
2. ASUE BHKW-Kenndaten (2011).

## Limitations
- Steady-state only
- No distinction between exhaust heat and jacket water heat recovery
- Single-engine model (no multi-engine dispatch)
- No humidity correction
