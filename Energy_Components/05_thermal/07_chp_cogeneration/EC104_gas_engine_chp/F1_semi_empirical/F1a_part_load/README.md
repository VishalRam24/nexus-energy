# EC104 — Gas Engine CHP — F1a Part-Load Efficiency Model

## Overview
Semi-empirical part-load model for a gas engine combined heat and power (CHP) system.
Polynomial correction curves map electrical and thermal efficiency as functions of part-load
ratio (PLR), following the ASHRAE/EPA CHP catalog methodology.

## Model Equations
```
eta_el  = eta_el_rated * (b0 + b1*PLR + b2*PLR^2)
eta_th  = eta_th_rated * (c0 + c1*PLR)
fuel    = P_el_rated * PLR / eta_el                 [kW, LHV basis]
P_el    = P_el_rated * PLR                          [kW_e]
Q_th    = fuel * eta_th                             [kW_th]
eta_tot = eta_el + eta_th
```

## Inputs
| Name | Unit | Range | Description |
|------|------|-------|-------------|
| part_load_ratio | - | [0.5, 1.0] | Fraction of rated electrical output |

## Outputs
| Name | Unit | Description |
|------|------|-------------|
| electrical_power_kw | kW_e | Electrical output |
| thermal_power_kw | kW_th | Useful heat output (jacket water + exhaust) |
| fuel_input_kw | kW | Fuel input on LHV basis |
| eta_electrical | - | Electrical efficiency |
| eta_thermal | - | Thermal efficiency |
| eta_total | - | Combined (electrical + thermal) efficiency |

## Parameters
| Parameter | Value | Unit | Source |
|-----------|-------|------|--------|
| P_el_rated | 2000 | kW_e | Typical 2 MW gas engine |
| eta_el_rated | 0.42 | - | EPA CHP Catalog |
| eta_th_rated | 0.43 | - | EPA CHP Catalog |
| PLR_min | 0.50 | - | Typical gas engine minimum |
| b0, b1, b2 | 0.1, 0.9, 0.0 | - | PLR correction coefficients |
| c0, c1 | 0.3, 0.7 | - | Thermal PLR correction coefficients |

## Sources
1. US EPA. (2017). *Catalog of CHP Technologies*. Section 2: Technology Characterization —
   Reciprocating Internal Combustion Engines.
2. ASUE. (2011). *BHKW-Kenndaten 2011*. Arbeitsgemeinschaft für sparsamen und umweltfreundlichen
   Energieverbrauch e.V.

## Limitations
- Linear and quadratic PLR correction only (no transient start/stop losses)
- No ambient temperature correction on power output
- No degradation model (efficiency assumed constant over time)
- Single fuel type (natural gas); does not account for biogas Wobbe index variation
