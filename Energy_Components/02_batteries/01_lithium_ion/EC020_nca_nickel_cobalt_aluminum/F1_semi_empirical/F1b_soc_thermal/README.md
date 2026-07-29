# EC020 -- NCA Battery -- F1b SOC-Thermal Model

## Model Description
Semi-empirical voltage model with temperature-dependent internal resistance via Arrhenius kinetics. NCA has the highest activation energy (28 kJ/mol) among the Li-ion chemistries modeled, making it the most temperature-sensitive.

## Equations
- **OCV(SOC):** 5th-order polynomial
- **R(T) = R_ref * exp(E_a/R * (1/T - 1/T_ref))**
- **V = OCV(SOC) - I * R(T)**
- **Q = I^2 * R(T) + I * T * dOCV/dT**
- **C(T) = C_ref * (1 + alpha_c * (T - T_ref))**

## Inputs
| Parameter | Unit | Range | Description |
|-----------|------|-------|-------------|
| soc | - | [0, 1] | State of charge |
| current | A | [-15, 15] | Current (positive=discharge) |
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
- R_ref = 0.022 Ohm (at 298.15 K)
- E_a = 28,000 J/mol (highest -- NCA most temp-sensitive)
- C_ref = 3.5 Ah
- alpha_c = 0.005 /K
- dOCV/dT = -0.0005 V/K

## Sources
- Tremblay & Dessaint (2009). IEEE Trans. Veh. Technol., 58(8), 3961-3969.
- Schuster et al. (2015). J. Power Sources, 286, 580-589.
- Viswanathan et al. (2010). J. Electrochem. Soc., 157(10), A1040-A1046.

## Limitations
- Single lumped thermal parameter
- Constant dOCV/dT
- No degradation (NCA is particularly susceptible to thermal degradation)
