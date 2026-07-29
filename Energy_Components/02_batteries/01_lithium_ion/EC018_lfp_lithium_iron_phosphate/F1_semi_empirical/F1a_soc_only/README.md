# EC018 — LFP Battery — F1a SOC-Only Voltage Model

## Overview
Simple semi-empirical model for LFP (Lithium Iron Phosphate) battery cells. LFP is characterized by a very flat voltage plateau (~3.3V) and excellent cycle life. Suitable for system-level sizing and techno-economic analysis.

## Inputs
| Name | Unit | Range | Description |
|------|------|-------|-------------|
| soc | - | [0, 1] | State of charge |
| current | A | [-17.5, 17.5] | Current (positive = discharge) |

## Outputs
| Name | Unit | Description |
|------|------|-------------|
| voltage | V | Terminal voltage |
| ocv | V | Open-circuit voltage |
| power | W | Electrical power |
| dsoc_dt | 1/s | SOC time derivative |

## Parameters
A123 ANR26650M1B: 3.3V nominal, 2.5Ah, R_int = 30 mOhm.

## Sources
1. Chen et al. (2020). J. Electrochem. Soc., 167, 080534.
2. A123 Systems ANR26650M1B datasheet.

## Limitations
- No temperature dependence
- No degradation / aging
- Constant internal resistance
- No hysteresis

## Usage
```python
from scripts.predict import ComponentModel
model = ComponentModel()
result = model.predict({"soc": 0.5, "current": 2.5})
```
