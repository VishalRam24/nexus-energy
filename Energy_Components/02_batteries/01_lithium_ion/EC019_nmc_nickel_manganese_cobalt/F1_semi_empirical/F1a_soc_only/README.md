# EC019 — NMC Battery — F1a SOC-Only Voltage Model

## Overview
Simple semi-empirical model for NMC (Nickel Manganese Cobalt) lithium-ion battery cells. Predicts terminal voltage as a function of state-of-charge and current only. Suitable for system-level sizing, techno-economic analysis, and quick parametric studies.

## Inputs
| Name | Unit | Range | Description |
|------|------|-------|-------------|
| soc | - | [0, 1] | State of charge (0 = empty, 1 = full) |
| current | A | [-25, 25] | Current (positive = discharge, negative = charge) |

## Outputs
| Name | Unit | Description |
|------|------|-------------|
| voltage | V | Terminal voltage |
| ocv | V | Open-circuit voltage |
| power | W | Electrical power (positive = discharge) |
| dsoc_dt | 1/s | SOC time derivative (for integration) |

## Equations
```
OCV(SOC) = a0 + a1*SOC + a2*SOC^2 + a3*SOC^3 + a4*SOC^4 + a5*SOC^5
V_terminal = OCV(SOC) - I * R_internal
P = V_terminal * I
dSOC/dt = -I / (C_nominal * 3600)
```

## Parameters
Samsung INR21700-50E (NMC811): 3.6V nominal, 5.0Ah, R_int = 35 mOhm.
OCV polynomial fitted to Chen et al. (2020) data.

## Valid Operating Range
- SOC: 0–100%
- Current: -25A to +25A (5C charge/discharge)
- Temperature effects NOT modeled in F1a (see F1b for thermal)

## Sources
1. Chen et al. (2020). "Development of Experimental Techniques for Parameterization of Multi-scale Lithium-ion Battery Models." J. Electrochem. Soc., 167, 080534.
2. Tremblay & Dessaint (2009). "Experimental Validation of a Battery Dynamic Model for EV Applications." World Electric Vehicle J., 3(2).

## Limitations
- No temperature dependence (isothermal at 25C)
- No degradation / aging effects
- Constant internal resistance (no SOC/current dependency)
- OCV polynomial may be inaccurate at SOC < 5% or SOC > 95%
- No hysteresis between charge/discharge OCV

## Usage
```python
from scripts.predict import ComponentModel
model = ComponentModel()
result = model.predict({"soc": 0.5, "current": 2.5})
print(result["voltage"])  # ~3.51 V
```
