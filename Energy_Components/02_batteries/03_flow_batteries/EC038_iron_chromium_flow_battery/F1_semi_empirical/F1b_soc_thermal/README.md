# EC038 Iron-Chromium Flow Battery (ICFB) — F1b SOC-Thermal Model

## Overview
Semi-empirical SOC + thermal model for an iron-chromium aqueous redox flow battery stack.
Extends F1a (isothermal Nernst + ohmic) with:
- Temperature-dependent standard potential: `E0(T) = E0_ref + dOCV_dT * (T - T_ref)`
- Arrhenius cell resistance: `R(T) = R_ref * exp(E_a/R * (1/T - 1/T_ref))`
- Reversible entropic heat: `Q_rev = I * N * T * dOCV/dT`
- Pump parasitic losses: `P_pump = k_pump * I^2`

## Chemistry
| Half-reaction | Standard potential |
|---|---|
| Fe3+ + e- <-> Fe2+ (positive) | +0.77 V vs SHE |
| Cr3+ + e- <-> Cr2+ (negative) | -0.41 V vs SHE |
| **Cell** | **1.18 V** |

n = 1 (single electron per formula unit)

## Inputs
| Name | Unit | Range | Notes |
|---|---|---|---|
| soc | dimensionless | 0.05 – 0.95 | Clamped internally to 0.01–0.99 |
| current | A | -100 – +100 | Positive = discharge |
| temperature | K | 288.15 – 338.15 | 15 to 65 degC |

## Outputs
| Name | Unit | Description |
|---|---|---|
| stack_voltage | V | N_cells * cell voltage |
| cell_voltage | V | Single cell terminal voltage |
| power | W | Net power (electrical minus pump losses) |
| heat_generation | W | Joule + reversible entropic heat |
| pump_loss | W | Parasitic pump power (I^2 scaling) |
| internal_resistance_cell | Ohm | Arrhenius R per cell |
| e_nernst | V | Nernst potential per cell |
| efficiency | dimensionless | Voltage efficiency |

## Key Parameters
| Parameter | Value | Source |
|---|---|---|
| N_cells | 40 | Generic stack |
| E0_ref | 1.18 V | Hruska & Savinell (1981) |
| R_cell_ref | 0.008 Ohm (4.0 Ohm.cm2 / 500 cm2) | Zeng et al. (2015) |
| E_a | 20,000 J/mol | Zeng et al. (2015); ICFB Cr kinetics |
| dOCV/dT | -0.0004 V/K/cell | Atkins (2014) thermodynamics; Shibata & Sumino (1988) |

## Temperature Range
15 – 65 degC. ICFB tolerates higher temperature than Zn-Br2 systems; Cr2+/Cr3+ kinetics become
unacceptably sluggish below 15 degC. Upper limit set by electrolyte stability.

## References
- Hruska, L. W., Savinell, R. F. (1981). Investigation of Factors Affecting Performance of the Iron-Redox Battery. *J. Electrochem. Soc.* 128(1), 18-25.
- Shibata, S., Sumino, M. P. (1988). A new vanadium redox flow battery. *J. Power Sources* 25, 177-184.
- Zeng, Y. K. et al. (2015). A comparative study of all-vanadium and iron-chromium redox flow batteries for large-scale energy storage. *J. Power Sources* 278, 294-302.
- Atkins, P. W. (2014). *Physical Chemistry*, 10th ed. Oxford. (Thermodynamic data for Fe/Cr couples)

## Limitations
- Isothermal Nernst model: activity coefficients assumed unity
- Pump loss modeled as simple I^2 scaling (actual scaling depends on electrolyte viscosity and system design)
- No degradation, membrane crossover, or self-discharge
- Single E_a for all resistance sources (membrane + electrode)
