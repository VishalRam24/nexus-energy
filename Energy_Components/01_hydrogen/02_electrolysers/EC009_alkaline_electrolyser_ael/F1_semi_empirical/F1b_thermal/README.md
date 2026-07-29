# EC009 -- Alkaline Electrolyser (AEL) -- F1b Thermal Model

## Overview
Temperature-dependent V-I model for alkaline water electrolysis with KOH electrolyte.
Models KOH conductivity, gas bubble coverage, and electrode kinetics as functions of temperature.

## Physics
- **E_rev(T)** = 1.229 - 0.000846*(T - 298.15)
- **KOH conductivity**: sigma(T) = sigma_ref * exp(-E_act/R*(1/T - 1/T_ref))
- **Bubble coverage**: theta(j,T) = bubble_coeff * (j/j_L)^0.3 * sqrt(T_ref/T)
- **Bruggeman correction**: sigma_eff = sigma_KOH * (1-theta)^1.5
- **Ohmic loss**: V_ohm = j * d_gap / sigma_eff
- **Activation**: V_act = RT/(alpha*nF) * arcsinh(j/(2*i0(T)))
- **V_cell** = E_rev + V_act + V_ohm

## Inputs
| Name | Unit | Range | Description |
|------|------|-------|-------------|
| current_density | A/m2 | 0 -- 5000 | Current density |
| temperature | K | 333 -- 373 | Cell temperature |
| koh_concentration | wt% | 25 -- 40 | KOH concentration (optional) |

## Outputs
| Name | Unit | Description |
|------|------|-------------|
| cell_voltage | V | Single cell voltage |
| power_consumption | kW | Stack power consumption |
| efficiency | - | Overall efficiency (LHV basis) |
| h2_production_rate | mol/s | H2 production rate (full stack) |

## Default Parameters
- sigma_KOH_ref = 0.5 S/cm at 30wt%/343K
- E_act_koh = 15000 J/mol, electrode_gap = 0.002 m

## References
- Ulleberg (2003), Int. J. Hydrogen Energy, 28(1), 21-33
- See & White (1997), J. Chem. Eng. Data, 42(6), 1266-1268
