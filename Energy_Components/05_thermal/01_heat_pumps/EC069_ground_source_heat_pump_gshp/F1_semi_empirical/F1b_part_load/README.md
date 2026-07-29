# EC069 — Ground-Source Heat Pump (GSHP) — F1b Part-Load

## Model

Extends F1a (Carnot-fraction COP) with:

1. **Part-load factor** (EN 14825): PLF = 1 - C_d * (1 - PLR), C_d = 0.20
2. **Seasonal ground temperature**: T_ground(month) = T_mean - A * cos(2*pi*(month - month_min)/12)

## Inputs

| Parameter | Unit | Range | Default |
|-----------|------|-------|---------|
| T_ground OR month | degC or month | 0-20 or 1-12 | 10 |
| T_sink | degC | 25 to 65 | - |
| part_load_ratio | - | 0 to 1 | 1.0 |

## Outputs

| Parameter | Unit |
|-----------|------|
| cop | dimensionless |
| heating_capacity_kw | kW |
| electrical_input_kw | kW |
| T_ground | degC |

## References

- EN 14825:2016
- ASHRAE Handbook HVAC Applications (2019), Ch. 34
- Staffell et al. (2012). Energy Environ. Sci., 5, 9291-9306
