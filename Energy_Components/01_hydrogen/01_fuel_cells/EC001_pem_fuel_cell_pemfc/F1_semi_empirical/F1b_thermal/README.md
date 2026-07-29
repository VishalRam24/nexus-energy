# EC001 -- PEM Fuel Cell (PEMFC) -- F1b Thermal Model

## Overview
Temperature-dependent semi-empirical polarization curve model for PEM fuel cells.
Extends F1a by making all loss mechanisms explicitly functions of temperature via
Arrhenius kinetics (exchange current density) and the Springer membrane conductivity model.

## Physics
- **Nernst potential**: E_rev(T) = 1.229 - 0.000846*(T - 298.15) + RT/(2F)*ln(pH2*sqrt(pO2))
- **Exchange current density**: i0(T) = i0_ref * exp(-E_act/R * (1/T - 1/T_ref))
- **Activation loss**: V_act = RT/(alpha*n*F) * arcsinh(j / (2*i0(T)))
- **Membrane conductivity**: sigma(T) = (0.005139*lambda - 0.00326) * exp(1268*(1/303 - 1/T))
- **Ohmic loss**: V_ohm = j * t_mem / sigma(T)
- **Concentration loss**: V_conc = -B * ln(1 - j/j_L)
- **Heat generation**: Q = j * (1.481 - V_cell)

## Inputs
| Name | Unit | Range | Description |
|------|------|-------|-------------|
| current_density | A/cm2 | 0 -- 2.0 | Current density |
| temperature | K | 333 -- 363 | Cell temperature |
| pressure_h2 | atm | 0.5 -- 3.0 | H2 partial pressure (optional) |
| pressure_o2 | atm | 0.1 -- 1.0 | O2 partial pressure (optional) |

## Outputs
| Name | Unit | Description |
|------|------|-------------|
| cell_voltage | V | Single cell voltage |
| power_density | W/cm2 | Electrical power per unit area |
| efficiency | - | Voltage efficiency (HHV basis) |
| heat_generation | W/cm2 | Waste heat per unit area |
| membrane_resistance | ohm cm2 | Temperature-dependent membrane ASR |

## Default Parameters
- i0_ref = 1e-4 A/cm2, E_act = 66000 J/mol
- sigma_ref = 0.1 S/cm, membrane_thickness = 0.0183 cm
- A_cell = 100 cm2, N_cells = 50

## References
- Amphlett et al. (1995), J. Electrochem. Soc., 142(1), 1-8
- Springer et al. (1991), J. Electrochem. Soc., 138(8), 2334-2342
