# EC040 Hydrogen-Bromine Flow Battery (HBrFB) — F1b SOC-Thermal Model

## Overview
Semi-empirical SOC + thermal model for a hydrogen-bromine aqueous flow battery stack.
Extends F1a with temperature-dependent resistance (Nafion proton conductivity), Nernst thermal
factor, entropic heat, and pump losses (electrolyte + H2 recirculation).

## Chemistry
| Half-reaction | Standard potential |
|---|---|
| Br2 + 2e- <-> 2Br- (positive) | +1.065 V vs SHE |
| 2H+ + 2e- <-> H2(g) (negative) | 0.000 V vs SHE |
| **Cell** | **~1.09 V** |

n = 2 (two-electron bromine reduction). H2 side uses Pt/C catalyst for fast kinetics.

## Inputs
| Name | Unit | Range | Notes |
|---|---|---|---|
| soc | dimensionless | 0.05 – 0.95 | Clamped to 0.01–0.99 internally |
| current | A | -150 – +150 | Positive = discharge; high I feasible due to fast H2/Pt kinetics |
| temperature | K | 293.15 – 333.15 | 20 to 60 degC |

## Outputs
| Name | Unit | Description |
|---|---|---|
| stack_voltage | V | N_cells * cell voltage |
| cell_voltage | V | Single cell terminal voltage |
| power | W | Net power (electrical minus pump losses) |
| heat_generation | W | Joule + reversible entropic heat |
| pump_loss | W | Parasitic pump power (I^2 scaling, includes H2 recirculation) |
| internal_resistance_cell | Ohm | Arrhenius R per cell (Nafion-dominated) |
| e_nernst | V | Nernst potential per cell |
| efficiency | dimensionless | Voltage efficiency |

## Key Parameters
| Parameter | Value | Source |
|---|---|---|
| N_cells | 30 | Generic stack |
| E0_ref | 1.09 V | Livshits et al. (2006); Cho et al. (2012) |
| R_cell_ref | 0.0125 Ohm (2.5 Ohm.cm2 / 200 cm2) | Cho et al. (2012) |
| E_a | 14,000 J/mol | Nafion proton conductivity; Cho et al. (2012); Tucker et al. (2015) |
| dOCV/dT | -0.0005 V/K/cell | Livshits et al. (2006); Atkins (2014) thermodynamics |

## Temperature Range
20 – 60 degC. Below 20 degC: Br2 dissolution limited and H2 kinetics slow. Above 60 degC:
Nafion membrane degradation and Br2 vapor pressure issues.

## References
- Livshits, V. et al. (2006). Advanced H2-Br2 fuel cell with a boron-doped diamond electrode. *J. Power Sources* 160, 1298-1301.
- Cho, K. T. et al. (2012). High performance hydrogen/bromine redox flow battery for grid-scale energy storage. *J. Electrochem. Soc.* 159(11), A1806-A1815.
- Tucker, M. C. et al. (2015). Performance of metal bipolar plates in the hydrogen-bromine redox flow battery. *J. Electrochem. Soc.* 162(8), A2159-A2165.
- Atkins, P. W. (2014). *Physical Chemistry*, 10th ed. Oxford. (HBr formation thermodynamics)

## Limitations
- Nafion single-E_a model; actual proton conductivity has more complex T-dependence
- No Br2 crossover through membrane
- H2 side overpotential assumed negligible (fast Pt/C kinetics)
- Pump loss as simple I^2 scaling
