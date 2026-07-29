# EC091 — Vapor Compression Chiller — F1b Part-Load (IPLV)

## Model

Extends F1a with DOE-2/AHRI IPLV methodology:

```
EIR_fPLR = d1 + d2*PLR + d3*PLR^2    (DOE-2 chiller curve)
f_T      = e1 + e2*T_cw + e3*T_cw^2  (condenser temp correction)
COP      = COP_ref * f_T / EIR_fPLR
IPLV     = 1 / (0.01/COP_100 + 0.42/COP_75 + 0.45/COP_50 + 0.12/COP_25)
```

## Inputs

| Parameter | Unit | Range | Default |
|-----------|------|-------|---------|
| T_chw | degC | 4-12 | 6.7 |
| T_cw | degC | 15-45 | 29.4 |
| PLR | - | 0-1 | 1.0 |

## Outputs

| Parameter | Unit |
|-----------|------|
| cop | dimensionless |
| cooling_capacity_kw | kW |
| electrical_input_kw | kW |
| iplv | dimensionless |

## References

- AHRI Standard 550/590
- DOE-2 Reference Manual
- EnergyPlus Chiller:Electric:EIR
- Gordon & Ng (2000), Cool Thermodynamics
