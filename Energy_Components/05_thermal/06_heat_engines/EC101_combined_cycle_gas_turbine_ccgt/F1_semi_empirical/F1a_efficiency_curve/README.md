# EC101 — Combined Cycle Gas Turbine (CCGT) — F1a Efficiency Curve

## Model Overview

Semi-empirical part-load efficiency model corrected for ambient temperature.

```
eta(PLR, T_amb) = eta_iso * f_PLR(PLR) * f_amb(T_amb)

f_PLR(PLR) = a0 + a1*PLR + a2*PLR^2      (quadratic part-load factor)
f_amb(T_amb) = 1 - k_amb*(T_amb - T_ref) (linear ambient derating)

P_out           = P_rated * PLR
fuel_rate       = P_out / eta  [MW_th] / LHV  → kg/s
exhaust_temp    = 80 + 520 * PLR^0.3  [degC]
```

## Parameters

| Parameter | Value | Unit | Source |
|-----------|-------|------|--------|
| P_rated   | 571   | MW_e | Kehlhofer et al. (2009) |
| eta_iso   | 0.64  | –    | Net LHV, ISO conditions |
| T_amb_ref | 15    | degC | ISO 2314 |
| k_amb     | 0.005 | 1/K  | Typical CCGT derating |
| a0        | 0.15  | –    | Part-load polynomial |
| a1        | 0.85  | –    | Part-load polynomial |
| a2        | 0.00  | –    | Part-load polynomial |
| LHV_gas   | 50    | MJ/kg | Natural gas |

## Inputs / Outputs

**Inputs:**

| Name | Unit | Valid Range |
|------|------|-------------|
| part_load_ratio | – | 0.30 – 1.00 |
| ambient_temp    | degC | –20 – 50 |

**Outputs:**

| Name | Unit |
|------|------|
| power_mw       | MW_e |
| efficiency     | – (LHV) |
| fuel_rate_kgs  | kg/s |
| exhaust_temp_c | degC |

## Usage

```python
from scripts.predict import ComponentModel
model = ComponentModel()
result = model.predict({"part_load_ratio": 0.8, "ambient_temp": 30.0})
print(result)
```

## Reference

Kehlhofer, R., Hannemann, F., Stirnimann, F., Rukes, B. (2009).
*Combined-Cycle Gas & Steam Turbine Power Plants*, 3rd ed. PennWell Corporation.

## Limitations

- Linear ambient correction is valid for –20 to 50 degC; wider ranges require nonlinear terms.
- Exhaust temperature model is empirical (±20 degC accuracy vs. detailed cycle simulation).
- Degradation effects (fouling, aging) are not included in this fidelity level.
