# EC010 -- Solid Oxide Electrolyser (SOEC) -- F1b Thermal Model

## Overview
Temperature-dependent V-I model for solid oxide electrolysis cells.
Very strong T-dependence (973-1123 K). Models endothermic/thermoneutral/exothermic
operation regimes based on cell voltage relative to thermoneutral voltage.

## Physics
- **E_rev(T)** = 1.253 - 0.00024*(T - 298.15)
- **E_tn(T)** = 1.285 - 0.000065*(T - 298.15)
- **YSZ conductivity**: sigma_ion(T) = (A/T)*exp(-E_act/(RT))
- **Ohmic ASR**: R_ohm = t_elec / sigma_ion(T)
- **Activation**: V_act = RT/(alpha*nF)*[arcsinh(j/(2*i0_a(T))) + arcsinh(j/(2*i0_c(T)))]
- **V_cell** = E_rev + V_act + V_ohm
- **Thermal mode**: V < E_tn = endothermic, V > E_tn = exothermic

## Inputs
| Name | Unit | Range | Description |
|------|------|-------|-------------|
| current_density | A/cm2 | 0 -- 2.5 | Current density |
| temperature | K | 973 -- 1123 | Cell temperature |
| steam_utilization | - | 0 -- 0.8 | Steam utilization (optional) |

## Outputs
| Name | Unit | Description |
|------|------|-------------|
| cell_voltage | V | Single cell voltage |
| power_consumption | W/cm2 | Electrical power input per area |
| efficiency | - | Thermoneutral efficiency |
| h2_production_rate | mol/s/cm2 | H2 production rate |
| thermal_mode | - | endothermic/thermoneutral/exothermic |

## Default Parameters
- Same YSZ ionic conductivity model as SOFC (A_sigma=3.34e4, E_act_ion=80000)
- Reversed electrode kinetics for electrolysis mode

## References
- Ni et al. (2007), Chem. Eng. Tech., 29(6), 636-642
- Udagawa et al. (2007), J. Power Sources, 166(1), 127-136
- Kazempoor & Braun (2014), Int. J. Hydrogen Energy, 39(5), 2669-2684
