# EC028 -- Lead-Acid Battery -- F1b SOC-Thermal Model

## Model Description
Semi-empirical voltage model for a 12V/100Ah flooded lead-acid battery with temperature-dependent internal resistance via Arrhenius kinetics. Lead-acid has strong capacity-temperature dependence (alpha_c = 0.01/K) and includes Peukert exponent for rate-dependent capacity.

## Equations
- **OCV(SOC):** 3rd-order polynomial (6-cell, 12V nominal)
- **R(T) = R_ref * exp(E_a/R * (1/T - 1/T_ref))**
- **V = OCV(SOC) - I * R(T)**
- **Q = I^2 * R(T) + I * T * 6 * dOCV_dT_per_cell**
- **C(T) = C_ref * (1 + alpha_c * (T - T_ref))**

## Inputs
| Parameter | Unit | Range | Description |
|-----------|------|-------|-------------|
| soc | - | [0, 1] | State of charge |
| current | A | [-50, 50] | Current (positive=discharge) |
| temperature | K | [253.15, 333.15] | Battery temperature |

## Outputs
| Parameter | Unit | Description |
|-----------|------|-------------|
| terminal_voltage | V | Terminal voltage |
| power | W | Electrical power |
| heat_generation | W | Total heat generation |
| effective_capacity | Ah | Temperature-corrected capacity |
| internal_resistance | Ohm | Temperature-dependent resistance |

## Key Parameters
- R_ref = 0.008 Ohm (at 298.15 K)
- E_a = 15,000 J/mol (lower than Li-ion)
- C_ref = 100.0 Ah (C20 rate)
- alpha_c = 0.01 /K (strong temperature dependence)
- Peukert exponent n = 1.2
- dOCV/dT = -0.0003 V/K per cell

## Sources
- Copetti et al. (1993). Progress in Photovoltaics, 1(4), 283-292.
- Manwell & McGowan (1993). Solar Energy, 50(5), 399-405.
- Bode (1977). Lead-Acid Batteries, Wiley.
- Schiffer et al. (2007). J. Power Sources, 168, 66-78.

## Limitations
- Peukert effect stored but not yet applied to dynamic capacity
- No stratification or sulfation modeling
- No gassing/water loss effects
