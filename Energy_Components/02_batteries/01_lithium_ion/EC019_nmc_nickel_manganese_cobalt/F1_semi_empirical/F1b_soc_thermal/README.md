# EC019 -- NMC Battery -- F1b SOC-Thermal Model

## Model Description
Semi-empirical voltage model with temperature-dependent internal resistance via Arrhenius kinetics. Extends F1a (SOC-only) by adding thermal effects on resistance, capacity, and heat generation.

## Equations
- **OCV(SOC):** 5th-order polynomial (same as F1a)
- **R(T) = R_ref * exp(E_a/R * (1/T - 1/T_ref))** -- Arrhenius kinetics
- **V = OCV(SOC) - I * R(T)** -- Terminal voltage
- **Q = I^2 * R(T) + I * T * dOCV/dT** -- Heat generation (irreversible + reversible)
- **C(T) = C_ref * (1 + alpha_c * (T - T_ref))** -- Capacity correction

## Inputs
| Parameter | Unit | Range | Description |
|-----------|------|-------|-------------|
| soc | - | [0, 1] | State of charge |
| current | A | [-25, 25] | Current (positive=discharge) |
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
- R_ref = 0.025 Ohm (at 298.15 K)
- E_a = 25,000 J/mol
- C_ref = 5.0 Ah
- alpha_c = 0.005 /K
- dOCV/dT = -0.0004 V/K

## Sources
- Chen et al. (2020). J. Electrochem. Soc., 167, 080534.
- Ecker et al. (2015). J. Electrochem. Soc., 162(9), A1836-A1848.
- Forgez et al. (2010). J. Power Sources, 195(9), 2961-2968.

## Limitations
- Single lumped thermal parameter (no spatial temperature distribution)
- Constant dOCV/dT (in reality varies with SOC)
- No degradation or aging effects
- No hysteresis between charge/discharge OCV
