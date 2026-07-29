# EC018 -- LFP Battery -- F1b SOC-Thermal Model

## Model Description
Semi-empirical voltage model with temperature-dependent internal resistance via Arrhenius kinetics. LFP has lower activation energy (22 kJ/mol) than NMC/NCA, making it less temperature-sensitive.

## Equations
- **OCV(SOC):** 5th-order polynomial with characteristic flat plateau ~3.3V
- **R(T) = R_ref * exp(E_a/R * (1/T - 1/T_ref))**
- **V = OCV(SOC) - I * R(T)**
- **Q = I^2 * R(T) + I * T * dOCV/dT**
- **C(T) = C_ref * (1 + alpha_c * (T - T_ref))**

## Inputs
| Parameter | Unit | Range | Description |
|-----------|------|-------|-------------|
| soc | - | [0, 1] | State of charge |
| current | A | [-17.5, 17.5] | Current (positive=discharge) |
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
- R_ref = 0.020 Ohm (at 298.15 K)
- E_a = 22,000 J/mol (lower than NMC -- LFP less temp-sensitive)
- C_ref = 3.3 Ah
- alpha_c = 0.004 /K
- dOCV/dT = -0.0002 V/K (LFP smaller entropy change)

## Sources
- Chen et al. (2020). J. Electrochem. Soc., 167, 080534.
- Ecker et al. (2015). J. Electrochem. Soc., 162(9), A1836-A1848.
- Thomas et al. (2008). J. Power Sources, 184(2), 666-671.

## Limitations
- Single lumped thermal parameter
- Constant dOCV/dT (in reality varies significantly with SOC for LFP)
- No degradation or aging effects
