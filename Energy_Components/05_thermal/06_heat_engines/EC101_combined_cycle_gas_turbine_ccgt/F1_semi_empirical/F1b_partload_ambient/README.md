# EC101 -- Combined Cycle Gas Turbine (CCGT) -- F1b Part-Load + Ambient

## Overview
Extends F1a by modelling GT and ST bottoming cycle separately. GT has its own part-load curve and ambient correction; ST efficiency depends on GT exhaust heat availability. Combined efficiency = eta_GT + (1-eta_GT)*eta_ST_eff.

## Model Equations
```
GT efficiency:
    eta_GT = eta_GT_rated * (a + b*PLR + c*PLR^2) * sqrt(T_ref/T_amb)

GT power correction:
    P_GT = P_rated * PLR * (P_amb/P_ref) * sqrt(T_ref/T_amb)

ST (bottoming) efficiency:
    eta_ST_eff(PLR) = eta_ST_rated * (st_a + st_b * PLR)

Combined efficiency:
    eta_cc = eta_GT + (1 - eta_GT) * eta_ST_eff

Heat rate:
    HR = 3600 / eta_cc [kJ/kWh]
```

## Inputs
| Name | Unit | Range | Default |
|------|------|-------|---------|
| PLR | - | [0.4, 1.0] | - |
| T_ambient | K | [243, 323] | 288.15 |
| P_ambient | kPa | [80, 110] | 101.325 |

## Outputs
| Name | Unit | Description |
|------|------|-------------|
| efficiency_combined | - | Net combined efficiency |
| efficiency_gt | - | GT efficiency |
| efficiency_st | - | ST effective efficiency |
| power_output_kw | kW | Combined output |
| heat_rate_kj_kwh | kJ/kWh | Combined heat rate |
| exhaust_temp_K | K | Stack temperature |

## Key Physics
- GT exhaust temperature increases slightly at part load (maintains HRSG performance)
- Combined efficiency ~62% at full load, ~55% at 50% load (typical F-class CCGT)
- Ambient temperature affects GT air density (power and efficiency)
- ST performance degrades at part load due to reduced exhaust heat

## Sources
1. Kehlhofer et al. (2009), Combined-Cycle Gas & Steam Turbine Power Plants, 3rd ed.
2. Chase, D.L. (2001), Combined-Cycle Development, Evolution, and Future.

## Limitations
- Single-shaft 2x1 configuration assumed
- No duct firing or supplementary firing
- No steam turbine bypass modelling
- Steady-state only
