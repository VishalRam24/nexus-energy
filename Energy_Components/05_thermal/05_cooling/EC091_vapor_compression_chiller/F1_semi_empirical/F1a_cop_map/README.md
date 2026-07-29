# EC091 — Vapor Compression Chiller — F1a COP Map Model

## Overview
Carnot-fraction COP model for a vapor compression chiller with Gordon-Ng part-load polynomial.
COP = eta_Carnot * T_evap / (T_cond - T_evap), part-load adjusted as COP(PLR) = COP_full * (c1 + c2*PLR + c3*PLR^2).

## Inputs
| Name | Unit | Range | Description |
|------|------|-------|-------------|
| T_chw_supply | degC | [4, 12] | Chilled water supply / evaporator temperature |
| T_cond | degC | [25, 45] | Condenser temperature (cooling tower supply) |
| part_load_ratio | - | [0.1, 1.0] | Part-load ratio (default 1.0) |

## Outputs
| Name | Unit | Description |
|------|------|-------------|
| cop | - | Coefficient of performance (cooling) |
| cooling_kw | kW | Cooling output |
| electrical_kw | kW | Compressor electrical input |
| heat_rejection_kw | kW | Heat rejected to condenser (Q_cool + W_comp) |

## Parameters
- Q_rated: 500 kW cooling
- COP_rated: 5.5 at T_evap=5C, T_cond=35C
- Carnot fraction: 0.55
- Refrigerant: R134a
- PLR polynomial: c1=0.1, c2=0.9, c3=0.0

## Sources
1. Gordon, J.M. & Ng, K.C. (2000). Cool Thermodynamics. Cambridge International Science Publishing.
2. ASHRAE Handbook — HVAC Systems and Equipment (2020).

## Limitations
- No refrigerant property variation (CoolProp not used at F1a)
- No compressor speed variation (constant-speed assumed)
- No heat exchanger fouling or degradation
- PLR range limited to 0.1–1.0 (model not valid below minimum part-load)
