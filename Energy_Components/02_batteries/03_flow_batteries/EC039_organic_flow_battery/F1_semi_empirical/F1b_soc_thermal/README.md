# EC039 Organic Flow Battery (OFB) — F1b SOC-Thermal Model

## Overview
Semi-empirical SOC + thermal model for an aqueous organic redox flow battery stack.
Representative chemistry: AQDS (anthraquinone-2,7-disulphonate) / ferricyanide.
Extends F1a with temperature-dependent resistance, Nernst thermal factor, and pump losses.

## Chemistry (AQDS/Ferricyanide representative)
| Half-reaction | Standard potential |
|---|---|
| Fe(CN)6^3- + e- <-> Fe(CN)6^4- (positive) | ~+0.36 V vs SHE |
| AQDS + 2H+ + 2e- <-> H2AQDS (negative) | ~-0.33 V vs SHE |
| **Cell** | **~0.70 V** |

n = 2 (two-electron quinone reduction)

## Inputs
| Name | Unit | Range | Notes |
|---|---|---|---|
| soc | dimensionless | 0.05 – 0.95 | Clamped to 0.01–0.99 internally |
| current | A | -50 – +50 | Positive = discharge; small 20-cell stack |
| temperature | K | 283.15 – 313.15 | 10 to 40 degC |

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
| N_cells | 20 | Generic small stack |
| E0_ref | 0.70 V | Huskinson et al. (2014); AQDS/ferricyanide |
| R_cell_ref | 0.06 Ohm (6.0 Ohm.cm2 / 100 cm2) | Kwabi et al. (2020) |
| E_a | 16,000 J/mol | Huskinson et al. (2014); Kwabi et al. (2020) |
| dOCV/dT | -0.0003 V/K/cell | Kwabi et al. (2020); Yang et al. (2011) |

## Temperature Range
10 – 40 degC. Upper limit set by degradation of organic active species (quinone
autoxidation and ring-opening accelerate above ~40-50 degC). Lower limit set by
electrolyte viscosity and ion transport.

## References
- Huskinson, B. et al. (2014). A metal-free organic-inorganic aqueous flow battery. *Nature* 505, 195-198.
- Lin, K. et al. (2015). Alkaline quinone flow battery. *Science* 349, 1529-1532.
- Janoschka, T. et al. (2015). An aqueous, polymer-based redox-flow battery using non-corrosive, safe, and low-cost materials. *Nature* 527, 78-81.
- Kwabi, D. G. et al. (2020). Electrolyte lifetime in aqueous organic flow batteries: A critical review. *Chem. Rev.* 120, 6467-6489.
- Yang, Z. et al. (2011). Electrochemical energy storage for green grid. *Chem. Sus. Chem.* 4, 1338-1345.

## Limitations
- Single representative chemistry (AQDS/ferricyanide); different organic couples will have different E0 and dOCV/dT
- No molecular degradation kinetics modeled
- No crossover or self-discharge
- Activity coefficients assumed unity (dilute solution approximation)
