# EC074 — Plate Heat Exchanger — F1b Fouling

## Model

Extends F1a (e-NTU counter-flow) with fouling resistance:

```
1/U_fouled = 1/U_clean + Rf_hot + Rf_cold
NTU = U_fouled * A / C_min
```

The fouled NTU is used in the standard counter-flow effectiveness formula.

## Inputs

| Parameter | Unit | Range | Default |
|-----------|------|-------|---------|
| T_h_in | degC | 30-120 | - |
| T_c_in | degC | 5-60 | - |
| m_dot_hot | kg/s | 0.05-5.0 | - |
| m_dot_cold | kg/s | 0.05-5.0 | - |
| fouling_resistance_hot | m2K/W | 0-0.01 | 0.0001 |
| fouling_resistance_cold | m2K/W | 0-0.01 | 0.0001 |

## Outputs

| Parameter | Unit |
|-----------|------|
| Q_kw | kW |
| T_h_out, T_c_out | degC |
| effectiveness | - |
| ntu | - |
| U_fouled | W/m2K |
| effectiveness_reduction | - |

## References

- Incropera & DeWitt (2006), ch. 11
- TEMA Standards, 10th ed.
