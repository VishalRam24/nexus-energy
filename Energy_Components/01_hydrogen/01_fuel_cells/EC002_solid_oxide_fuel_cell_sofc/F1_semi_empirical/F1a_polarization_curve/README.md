# EC002 — Solid Oxide Fuel Cell (SOFC) — F1a Polarization Curve

## Overview
Semi-empirical Butler-Volmer polarization curve model for a hydrogen-fueled, anode-supported SOFC stack with YSZ electrolyte. Captures activation, ohmic, and concentration overpotentials.

## Model Equations
```
E_Nernst = E0 + (RT/2F) * ln(pH2 * sqrt(pO2) / pH2O)
V_act    = (RT / alpha*F) * arcsinh(j / (2*j0(T)))     [Butler-Volmer simplified]
V_ohm    = j * ASR
V_conc   = -(RT/nF) * ln(1 - j/j_L)
V_cell   = E_Nernst - V_act - V_ohm - V_conc
V_stack  = N_cells * V_cell
eta      = V_cell * n * F / LHV_H2
```
Temperature dependence: j0 scales linearly with T/T_op (first-order Arrhenius approximation at F1 fidelity).

## Inputs
| Name | Unit | Range | Description |
|------|------|-------|-------------|
| current_density | A/cm² | [0, 1.8] | Anode current density |
| temperature | degC | [600, 1000] | Cell operating temperature (default 800°C) |

## Outputs
| Name | Unit | Description |
|------|------|-------------|
| cell_voltage | V | Single-cell terminal voltage |
| stack_voltage | V | Full stack voltage (N_cells * V_cell) |
| power_density | W/cm² | Cell power density |
| stack_power_kw | kW | Stack electrical output |
| efficiency | — | Thermodynamic (LHV) efficiency |

## Design Parameters
| Parameter | Value | Unit |
|-----------|-------|------|
| T_op | 1073 | K (800°C) |
| N_cells | 40 | — |
| A_cell | 100 | cm² |
| ASR | 0.3 | Ω·cm² |
| j0 | 0.4 | A/cm² |
| j_L | 2.0 | A/cm² |
| alpha | 0.5 | — |
| pH2/pO2/pH2O | 0.97/0.21/0.03 | atm |

## Sources
1. Chan, S.H., Ho, H.K., Tian, Y. (2001), "Modelling of simple hybrid solid oxide fuel cell and gas turbine power plant", J. Power Sources, 93, 130-140.
2. Larminie, J. & Dicks, A. (2003), "Fuel Cell Systems Explained", 2nd ed., Wiley.

## Limitations
- j0 temperature scaling is linear (first-order); full Arrhenius would be F1b
- Constant partial pressures (no fuel utilization feedback)
- No thermal model (isothermal at specified T)
- No degradation or carbon deposition modeling
