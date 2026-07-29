# EC068 — Air-Source Heat Pump — F1a COP Map Model

## Overview
Carnot-fraction COP model for ASHP. COP = eta_Carnot * T_sink/(T_sink - T_source).

## Inputs
| Name | Unit | Range | Description |
|------|------|-------|-------------|
| T_source | degC | [-20, 40] | Outdoor air temperature |
| T_sink | degC | [25, 65] | Heating supply temperature |
| part_load_ratio | - | [0, 1] | Part-load ratio (default 1.0) |

## Outputs
| Name | Unit | Description |
|------|------|-------------|
| cop | - | Coefficient of performance |
| heating_capacity_kw | kW | Thermal output |
| electrical_input_kw | kW | Electrical consumption |

## Sources
1. Staffell et al. (2012). Energy Environ. Sci., 5, 9291-9306.
2. EN 14511 rating standard.

## Limitations
- No part-load COP degradation, no defrost cycle, no degradation
