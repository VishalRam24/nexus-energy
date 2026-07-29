# EC031 -- Sodium-Ion Battery -- F1b SOC-Thermal Model

## Model Description
Semi-empirical voltage model with temperature-dependent internal resistance via Arrhenius kinetics for Na-ion chemistry. Na-ion has slightly higher temperature sensitivity in capacity (alpha_c = 0.006/K) and moderate activation energy.

## Equations
- **OCV(SOC):** 5th-order polynomial (lower voltage range than Li-ion, ~2.2-3.8V)
- **R(T) = R_ref * exp(E_a/R * (1/T - 1/T_ref))**
- **V = OCV(SOC) - I * R(T)**
- **Q = I^2 * R(T) + I * T * dOCV/dT**
- **C(T) = C_ref * (1 + alpha_c * (T - T_ref))**

## Inputs
| Parameter | Unit | Range | Description |
|-----------|------|-------|-------------|
| soc | - | [0, 1] | State of charge |
| current | A | [-30, 30] | Current (positive=discharge) |
| temperature | K | [253.15, 333.15] | Cell temperature |

## Outputs
| Parameter | Unit | Description |
|-----------|------|-------------|
| terminal_voltage | V | Terminal voltage |
| power | W | Electrical power |
| heat_generation | W | Total heat generation |
| effective_capacity | Ah | Temperature-corrected capacity |
| internal_resistance | Ohm | Temperature-dependent resistance |

## Key Parameters
- R_ref = 0.030 Ohm (at 298.15 K)
- E_a = 24,000 J/mol
- C_ref = 3.0 Ah
- alpha_c = 0.006 /K
- dOCV/dT = -0.0003 V/K

## Sources
- Tremblay & Dessaint (2009). IEEE Trans. Veh. Technol., 58(8), 3961-3969.
- Hwang et al. (2017). Chem. Soc. Rev., 46, 3529.
- Rudola et al. (2021). J. Mater. Chem. A, 9, 8279.

## Limitations
- Limited experimental validation data for Na-ion thermal behavior
- Thermal parameters estimated from early Na-ion literature
- No degradation or aging effects
