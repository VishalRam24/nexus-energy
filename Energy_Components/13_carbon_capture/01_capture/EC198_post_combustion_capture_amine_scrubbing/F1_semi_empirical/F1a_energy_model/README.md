# EC198 — Post-Combustion Capture (Amine Scrubbing) — F1a Energy Model

## Overview
Energy consumption model for 30 wt% MEA-based post-combustion CO2 capture from power plant flue gas. The reboiler duty (dominant energy penalty) is modeled as a function of the liquid-to-gas (L/G) ratio, which itself depends on the required capture rate.

## Model Equations
```
LG(capture_rate) = LG_opt + (capture_rate - 0.90) * 4.0   [mol/mol]
q_reboiler       = q_base / (1 - exp(-k_LG * (LG - LG_min)))  [GJ/tCO2]
q_electricity    = 0.25 * (1 + 0.5*(capture_rate - 0.90))  [GJ/tCO2]
E_specific       = q_reboiler + q_electricity               [GJ/tCO2]

CO2_in           = flue_rate * x_CO2 * (MW_CO2/MW_flue)   [kg/s]
CO2_captured     = CO2_in * capture_rate                    [kg/s]

P_reboiler       = q_reboiler * CO2_captured                [MW]
P_electricity    = q_electricity * CO2_captured             [MW]
```
Parameters: q_base=3.2 GJ/tCO2, LG_opt=2.5, LG_min=1.5, k_LG=1.2

## Inputs
| Name | Unit | Range | Description |
|------|------|-------|-------------|
| flue_gas_rate | kg/s | [100, 1000] | Flue gas mass flow rate |
| co2_fraction | mol/mol | [0.04, 0.15] | CO2 mole fraction in flue gas |
| capture_rate | - | [0.80, 0.95] | Target CO2 capture fraction |

## Outputs
| Name | Unit | Description |
|------|------|-------------|
| co2_captured_kgs | kg/s | CO2 removed from flue gas |
| reboiler_duty_mw | MW | Thermal power to stripper reboiler |
| electricity_mw | MW | Electrical power demand |
| specific_energy_gjt | GJ/tCO2 | Total specific energy consumption |

## Design Point (500 kg/s flue gas, CO2=12%, capture=90%)
| Parameter | Value |
|-----------|-------|
| CO2 captured | ~28.5 kg/s (~2,470 t/day) |
| Reboiler duty | ~91 MW |
| Electricity | ~7.1 MW |
| Specific energy | ~3.45 GJ/tCO2 |

## Sources
1. Abu-Zahra, M.R.M., Schneiders, L.H.J., Niederer, J.P.M., Feron, P.H.M., Versteeg, G.F. (2007). CO2 capture from power plants: Part I. A parametric study of the technical performance based on monoethanolamine. *International Journal of Greenhouse Gas Control*, 1(1), 37–46.

## Limitations
- Fixed MEA concentration (30 wt%); other solvents (MDEA, piperazine) require different parameters
- No degradation, foaming, or corrosion effects
- Simplified L/G vs capture-rate relationship; real systems require rigorous column simulation
- Compression to pipeline pressure not included in electricity estimate
- Does not model partial-load or dynamic behavior
