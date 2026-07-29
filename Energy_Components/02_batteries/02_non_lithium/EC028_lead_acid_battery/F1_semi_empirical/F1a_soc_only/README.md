# EC028 — Lead-Acid Battery — F1a SOC-Voltage Model

## Model Overview

Modified Shepherd model: terminal voltage as a cubic OCV polynomial minus an ohmic drop.

```
OCV(SOC) = a0 + a1*SOC + a2*SOC^2 + a3*SOC^3
V        = OCV(SOC) - I * R_internal

dsoc/dt  = -I / (C * 3600)    [positive I = discharge]
power    = V * I               [W, positive = discharge]
```

## Parameters

| Parameter | Value | Unit | Note |
|-----------|-------|------|------|
| V_nom     | 12.0  | V    | Nominal pack voltage |
| C         | 100   | Ah   | Capacity |
| R_internal | 0.010 | Ohm | At 25 C |
| a0        | 11.5  | V    | OCV polynomial |
| a1        | 1.5   | V    | OCV polynomial |
| a2        | -0.5  | V    | OCV polynomial |
| a3        | 0.3   | V    | OCV polynomial |
| V_min     | 10.5  | V    | Discharge cutoff |
| V_max     | 14.4  | V    | Charge voltage limit |

## Inputs / Outputs

**Inputs:**

| Name | Unit | Range | Note |
|------|------|-------|------|
| soc | – | 0 – 1 | State of charge |
| current | A | –50 to 50 | Positive = discharge |

**Outputs:**

| Name | Unit |
|------|------|
| voltage  | V |
| ocv      | V |
| power    | W |
| dsoc_dt  | 1/s |

## Usage

```python
from scripts.predict import ComponentModel
model = ComponentModel()
result = model.predict({"soc": 0.8, "current": 20.0})
print(result)
```

## References

- Copetti, J.B., Lorenzo, E., Chenlo, F. (1993). *A general battery model for PV system simulation.* Progress in Photovoltaics, 1(4), 283-292.
- Manwell, J.F., McGowan, J.G. (1993). *Lead acid battery storage model for hybrid energy systems.* Solar Energy, 50(5), 399-405.

## Limitations

- Temperature dependence of R_internal and OCV is not included (F1b adds temperature).
- Peukert effect (capacity reduction at high rates) is not modeled.
- Self-discharge and electrolyte stratification are not included.
