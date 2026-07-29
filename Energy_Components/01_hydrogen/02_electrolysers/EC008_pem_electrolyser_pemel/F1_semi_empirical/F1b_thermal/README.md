# EC008 -- PEM Electrolyser (PEMEL) -- F1b Thermal Model

## Overview
Temperature-dependent V-I characteristic for PEM electrolysers.
Extends F1a with Arrhenius kinetics for both anode and cathode,
plus Springer membrane conductivity model.

## Physics
- **Thermoneutral voltage**: E_tn(T) = 1.481 - 0.000223*(T - 298)
- **Reversible voltage**: E_rev(T) = 1.229 - 0.000846*(T - 298.15)
- **Anode activation**: V_act_a = RT/(alpha*2F)*arcsinh(j/(2*i0_a(T)))
- **Cathode activation**: V_act_c = RT/(alpha*2F)*arcsinh(j/(2*i0_c(T)))
- **Exchange current**: i0(T) = i0_ref * exp(-E_act/R*(1/T - 1/T_ref))
- **Membrane**: sigma(T) = (0.005139*lambda - 0.00326)*exp(1268*(1/303 - 1/T))
- **V_cell** = E_rev + V_act_a + V_act_c + V_ohm
- **Heat**: Q = j*(V_cell - E_tn)

## Inputs
| Name | Unit | Range | Description |
|------|------|-------|-------------|
| current_density | A/cm2 | 0 -- 3.0 | Current density |
| temperature | K | 323 -- 363 | Cell temperature |
| pressure | bar | 1 -- 80 | Operating pressure (optional) |

## Outputs
| Name | Unit | Description |
|------|------|-------------|
| cell_voltage | V | Single cell voltage |
| power_consumption | W/cm2 | Electrical power input per area |
| efficiency_voltage | - | Thermoneutral efficiency |
| efficiency_faradaic | - | Faradaic (current) efficiency |
| h2_production_rate | mol/s/cm2 | H2 production rate |
| heat_generation | W/cm2 | Waste heat per unit area |

## Default Parameters
- i0_anode_ref = 1e-7 A/cm2, E_act_anode = 76000 J/mol
- sigma_ref = 0.1 S/cm, membrane_thickness = 0.0183 cm

## References
- Garcia-Valverde et al. (2012), Int. J. Hydrogen Energy, 37(2), 1927-1938
- Springer et al. (1991), J. Electrochem. Soc., 138(8), 2334-2342
