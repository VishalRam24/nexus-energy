# EC092 — Absorption Chiller — F1b Part-Load

## Model

Extends F1a (characteristic equation) with:

1. **Part-load COP**: COP_PL = COP_ref * (f1 + f2*PLR + f3*PLR^2)
2. **Driving heat correction**: f_Thot = g1 + g2*(T_hot - T_hot_rated)
3. **Crystallization protection**: PLR_min = 0.15

## Inputs

| Parameter | Unit | Range |
|-----------|------|-------|
| T_hot | degC | 70-120 |
| T_cw | degC | 25-45 |
| T_chw | degC | 4-15 |
| PLR | - | 0-1 |

## Outputs

| Parameter | Unit |
|-----------|------|
| cop | dimensionless |
| cooling_capacity_kw | kW |
| heat_input_kw | kW |

## References

- Herold, Radermacher & Klein (2016)
- Gordon & Ng (2000)
- ASHRAE Handbook (2020), Ch. 2
