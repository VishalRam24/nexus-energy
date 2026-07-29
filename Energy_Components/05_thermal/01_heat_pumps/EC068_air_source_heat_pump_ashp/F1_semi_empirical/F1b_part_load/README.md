# EC068 — Air-Source Heat Pump (ASHP) — F1b Part-Load

## Model

Extends F1a (Carnot-fraction COP map) with EN 14825 part-load degradation:

```
COP_full = eta_Carnot * T_sink / (T_sink - T_source)
PLF      = 1 - C_d * (1 - PLR)
COP_PL   = COP_full * PLF
```

Below PLR_min (typically 0.10), the compressor cycles on/off with additional
startup and transient losses.

## Inputs

| Parameter | Unit | Range | Default |
|-----------|------|-------|---------|
| T_source | degC | -20 to 40 | - |
| T_sink | degC | 25 to 65 | - |
| part_load_ratio | - | 0 to 1 | 1.0 |

## Outputs

| Parameter | Unit |
|-----------|------|
| cop | dimensionless |
| heating_capacity_kw | kW |
| electrical_input_kw | kW |
| cop_degradation_factor | dimensionless |

## Key Parameters

- **C_d = 0.25** — Degradation coefficient (EN 14825 default)
- **PLR_min = 0.10** — Minimum modulation ratio

## References

- EN 14825:2016 — Seasonal performance of heat pumps
- AHRI Standard 210/240
- Staffell et al. (2012). Energy Environ. Sci., 5, 9291-9306
