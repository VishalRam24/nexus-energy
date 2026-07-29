# EC020 — NCA Battery (Nickel-Cobalt-Aluminum) — F1a SOC-Only

## Model Card

| Field | Value |
|-------|-------|
| **EC ID** | EC020 |
| **Component** | NCA Li-ion Battery |
| **Cell** | Panasonic NCR18650B |
| **Fidelity** | F1a — SOC-only semi-empirical voltage model |
| **Framework** | Tremblay & Dessaint (2009) |
| **License** | BSD-3 |

## Physics

```
V_terminal = OCV(SOC) - I * R_internal
OCV(SOC)   = a0 + a1*SOC + a2*SOC^2 + a3*SOC^3 + a4*SOC^4 + a5*SOC^5
dSOC/dt    = -I / (C * 3600)
```

## OCV Polynomial Coefficients (NCA, Panasonic NCR18650B)

| Coeff | Value | Note |
|-------|-------|------|
| a0 | 2.7 | Intercept (~V at SOC=0) |
| a1 | 3.8 | Linear |
| a2 | -14.0 | Quadratic |
| a3 | 28.0 | Cubic |
| a4 | -25.0 | Quartic |
| a5 | 8.7 | Quintic |

OCV(0) ≈ 2.70 V, OCV(1) ≈ 4.20 V

## Cell Parameters

| Parameter | Value | Unit |
|-----------|-------|------|
| Capacity | 3.35 | Ah |
| V_nominal | 3.6 | V |
| V_max | 4.2 | V |
| V_min | 2.5 | V |
| R_internal | 0.045 | Ω |
| Max discharge | 15 A | (4.5C) |

## Inputs / Outputs

### Inputs
| Name | Unit | Range | Default |
|------|------|-------|---------|
| soc | — | 0–1 | — |
| current | A | -15 to 15 | 0 (positive=discharge) |

### Outputs
| Name | Unit | Description |
|------|------|-------------|
| voltage | V | Terminal voltage |
| ocv | V | Open-circuit voltage |
| power | W | Electrical power (positive=discharge) |
| dsoc_dt | 1/s | Rate of SOC change |

## Physics Checks (all pass)

- OCV monotonically increases with SOC
- Terminal voltage within [2.5, 4.2] V at all conditions
- Discharge voltage < OCV; charge voltage > OCV
- Power positive during discharge, negative during charge
- dSOC/dt negative during discharge, positive during charge
- R_internal ≥ 0.04 Ω (verified via voltage drop test)

## Data Sources

- Tremblay, O., & Dessaint, L.-A. (2009). Experimental validation of a battery dynamic model for EV applications. *World Electric Vehicle Journal*, 3(1), 289–298.
- Panasonic Corporation (2013). NCR18650B datasheet.

## Limitations

- No thermal dependence (isothermal, F1a only)
- No aging / capacity fade
- OCV polynomial may deviate from real cell below SOC=0.05 or above SOC=0.98
- Internal resistance assumed constant (no SOC or temperature dependence)
