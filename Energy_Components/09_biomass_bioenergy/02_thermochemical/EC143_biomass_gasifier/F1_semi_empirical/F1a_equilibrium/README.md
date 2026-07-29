# EC143 — Biomass Gasifier — F1a Equilibrium Model

## Overview
Simplified equilibrium model predicting dry syngas composition as a linear function of equivalence ratio (ER). Calibrated against published downdraft gasifier data for wood chips. Temperature correction applied for operation away from the 800°C design point.

## Model Equations
```
CO  = 0.22 - 0.15*(ER - 0.25)
H2  = 0.18 - 0.12*(ER - 0.25)
CO2 = 0.10 + 0.10*(ER - 0.25)
CH4 = 0.03 - 0.02*(ER - 0.25)
N2  = 1 - (CO + H2 + CO2 + CH4)   [balance]

LHV_syngas = 4.2*CO + 10.8*H2 + 35.8*CH4   [MJ/Nm3]

Cold Gas Efficiency = LHV_syngas * V_syngas / (m_biomass * LHV_biomass)
```

## Inputs
| Name | Unit | Range | Description |
|------|------|-------|-------------|
| equivalence_ratio | - | [0.2, 0.5] | Air/stoichiometric air ratio |
| temperature | degC | [700, 1000] | Gasification temperature |

## Outputs
| Name | Unit | Description |
|------|------|-------------|
| syngas_composition | mol_frac | Dict: CO, H2, CO2, CH4, N2 fractions (sum = 1.0) |
| lhv_syngas_mjnm3 | MJ/Nm3 | Lower heating value of dry syngas |
| cold_gas_efficiency | - | Ratio of syngas chemical energy to biomass input energy |

## Design Point (ER=0.25, T=800°C, Wood Chips)
| Species | Mole Fraction |
|---------|---------------|
| CO | 0.220 |
| H2 | 0.180 |
| CO2 | 0.100 |
| CH4 | 0.030 |
| N2 | 0.470 |
| **LHV** | **~5.0 MJ/Nm3** |
| **CGE** | **~0.70** |

## Sources
1. Zainal, Z.A., Ali, R., Lean, C.H., Seetharamu, K.N. (2001). Prediction of performance of a downdraft gasifier using equilibrium modeling for different biomass materials. *Energy Conversion and Management*, 42(12), 1499–1515.

## Limitations
- Linear ER sensitivity — valid only for 0.2 < ER < 0.5
- Fixed stoichiometry for wood chips (CH1.44O0.66); other feedstocks need parameter adjustment
- No tar, no char conversion, no moisture content effects
- Temperature correction is a minor empirical correction, not full equilibrium
- Steady-state only; no transient start-up or load-following dynamics
