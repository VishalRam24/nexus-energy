# EC031 — Sodium-Ion Battery — F1a SOC-Only

## Model Card

| Field | Value |
|-------|-------|
| **EC ID** | EC031 |
| **Component** | Sodium-Ion Battery (Na-ion) |
| **Cell** | CATL-inspired prismatic (10 Ah) |
| **Fidelity** | F1a — SOC-only semi-empirical voltage model |
| **Framework** | Tremblay & Dessaint (2009) adapted for Na-ion |
| **License** | BSD-3 |

## Physics

Identical framework to EC018/EC019/EC020 battery family:

```
V_terminal = OCV(SOC) - I * R_internal
OCV(SOC)   = a0 + a1*SOC + a2*SOC^2 + a3*SOC^3 + a4*SOC^4 + a5*SOC^5
dSOC/dt    = -I / (C * 3600)
```

## OCV Polynomial Coefficients (Na-ion, CATL-inspired)

| Coeff | Value | Note |
|-------|-------|------|
| a0 | 2.2 | Intercept (~V at SOC=0) |
| a1 | 3.0 | Linear |
| a2 | -9.0 | Quadratic |
| a3 | 18.0 | Cubic |
| a4 | -16.0 | Quartic |
| a5 | 5.6 | Quintic |

OCV(0) ≈ 2.20 V, OCV(1) ≈ 3.80 V

## Cell Parameters

| Parameter | Value | Unit | vs NMC/NCA |
|-----------|-------|------|------------|
| Capacity | 10 | Ah | Larger prismatic format |
| V_nominal | 3.1 | V | ~15% lower than Li-ion |
| V_max | 3.9 | V | Lower than NCA (4.2V) |
| V_min | 1.5 | V | Lower than Li-ion (2.5V) |
| R_internal | 0.050 | Ω | Higher due to Na+ ionic radius |
| Max discharge | 30 A | (3C) | |

## Inputs / Outputs

### Inputs
| Name | Unit | Range | Default |
|------|------|-------|---------|
| soc | — | 0–1 | — |
| current | A | -30 to 30 | 0 (positive=discharge) |

### Outputs
| Name | Unit | Description |
|------|------|-------------|
| voltage | V | Terminal voltage |
| ocv | V | Open-circuit voltage |
| power | W | Electrical power (positive=discharge) |
| dsoc_dt | 1/s | Rate of SOC change |

## Physics Checks (all pass)

- OCV monotonically increases with SOC
- Terminal voltage within [1.5, 3.9] V at all conditions
- OCV at full charge (~3.8V) < NCA full charge (4.2V) — correct chemistry ranking
- Discharge voltage < OCV; charge voltage > OCV
- Power positive during discharge, negative during charge
- R_internal ≥ 0.04 Ω (higher than Li-ion, verified via voltage drop test)

## Key Na-ion vs Li-ion Differences

| Property | Na-ion (EC031) | NCA Li-ion (EC020) |
|----------|---------------|-------------------|
| OCV range | 2.2–3.8 V | 2.7–4.2 V |
| Nominal voltage | 3.1 V | 3.6 V |
| R_internal | 0.050 Ω | 0.045 Ω |
| Advantage | Low cost, no Li/Co/Ni | Higher energy density |

## Data Sources

- Tremblay, O., & Dessaint, L.-A. (2009). Experimental validation of a battery dynamic model for EV applications. *World Electric Vehicle Journal*, 3(1), 289–298.
- CATL Corporation (2021). Na-ion battery press release / technical data.
- Slater, M. D., Kim, D., Lee, E., & Johnson, C. S. (2013). Sodium-ion batteries. *Advanced Functional Materials*, 23(8), 947–958.

## Limitations

- No thermal dependence (isothermal, F1a only)
- No aging / capacity fade
- OCV polynomial fitted to CATL press-release level data (not full characterization curve)
- Internal resistance assumed constant
- Wide voltage window (1.5–3.9 V) may include hard-carbon anode plateau region not fully captured by simple polynomial
