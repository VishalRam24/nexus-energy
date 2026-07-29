# EC079 -- Molten Salt TES -- F1b Stratified 10-Node Model

## Overview
10-node vertical stratification model for solar salt (60% NaNO3 + 40% KNO3) thermal energy storage. Builds on F1a (fully mixed) by resolving vertical temperature distribution, inter-node heat conduction, and temperature-dependent salt properties.

## Physics
- **Stratification**: 10 vertical control volumes, hot fluid enters top (charge), cold enters bottom (discharge)
- **Temperature-dependent properties**:
  - Density: rho(T) = 2090 - 0.636*T [kg/m3, T in degC]
  - Specific heat: cp(T) = 1443 + 0.172*T [J/(kg*K), T in degC]
- **Destratification**: Effective axial conductivity models turbulent mixing
- **Wall heat loss**: Per-node UA model with cylindrical geometry
- **Freezing constraint**: T > 220 degC enforced with warning at 230 degC

## Parameters
| Parameter | Value | Unit |
|-----------|-------|------|
| Tank volume | 1000 | m3 |
| Tank height | 14 | m |
| Number of nodes | 10 | - |
| U_wall | 0.5 | W/(m2*K) |
| k_destratification | 0.1 | W/(m*K) |
| T_hot_design | 565 | degC |
| T_cold_design | 290 | degC |
| T_freeze | 220 | degC |

## Inputs
| Name | Unit | Range |
|------|------|-------|
| T_charge_degC | degC | 290-600 |
| T_discharge_degC | degC | 220-565 |
| flow_rate_kg_s | kg/s | 0-2000 |
| mode | - | charge/discharge/idle |
| T_ambient_degC | degC | -20 to 60 |
| duration_s | s | 0-86400 |
| T_nodes_init | degC | array of 10 |

## Outputs
| Name | Unit |
|------|------|
| T_nodes | degC (array of 10) |
| T_outlet_degC | degC |
| stored_energy_kwh | kWh |
| thermal_efficiency | - |
| freeze_warning | bool |

## References
- Herrmann, U., Kelly, B., Price, H. (2004). Energy, 29(5-6), 883-893.
- Pacheco, J.E., Showalter, S.K., Kolb, W.J. (2002). ASME J. Solar Energy Eng. 124(2), 153-159.
- Zaversky, F. et al. (2013). Applied Energy, 109, 190-200.
