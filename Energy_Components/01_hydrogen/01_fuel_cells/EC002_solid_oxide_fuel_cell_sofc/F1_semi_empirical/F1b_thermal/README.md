# EC002 -- Solid Oxide Fuel Cell (SOFC) -- F1b Thermal Model

## Overview
Temperature-dependent semi-empirical polarization model for SOFCs.
Strong T-dependence (973-1273 K) modelled via Arrhenius YSZ ionic conductivity
and temperature-dependent electrode exchange current densities.

## Physics
- **Nernst**: E(T) = E0(T) + RT/(2F)*ln(pH2*sqrt(pO2)/pH2O)
- **YSZ conductivity**: sigma_ion(T) = (A_sigma/T)*exp(-E_act_ion/(RT))
- **Ohmic ASR**: R_ohm = t_elec / sigma_ion(T)
- **Activation**: V_act = RT/(alpha*n*F)*[arcsinh(j/(2*i0_a(T))) + arcsinh(j/(2*i0_c(T)))]
- **Concentration**: V_conc = -RT/(nF)*ln(1 - j/j_L)
- **Heat**: Q = j*(E_tn - V_cell)

## Inputs
| Name | Unit | Range | Description |
|------|------|-------|-------------|
| current_density | A/cm2 | 0 -- 2.0 | Current density |
| temperature | K | 973 -- 1273 | Cell temperature |
| fuel_utilization | - | 0 -- 0.9 | Fuel utilization (optional) |

## Outputs
| Name | Unit | Description |
|------|------|-------------|
| cell_voltage | V | Single cell voltage |
| power_density | W/cm2 | Electrical power per unit area |
| efficiency | - | Voltage efficiency (LHV basis) |
| asr | ohm cm2 | Total area-specific resistance |
| heat_generation | W/cm2 | Waste heat per unit area |

## Default Parameters
- A_sigma = 3.34e4 S K/cm, E_act_ion = 80000 J/mol
- i0_anode_ref = 0.5 A/cm2, E_act_anode = 100000 J/mol
- thickness_electrolyte = 0.001 cm

## References
- Chan et al. (2001), J. Power Sources, 93, 130-140
- Virkar (2005), J. Power Sources, 147(1-2), 125-136
