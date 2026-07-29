# EC198 — Post-Combustion Capture (Amine Scrubbing) — F1b Part-Load + Degradation Model

## Overview
Extends the F1a energy model with part-load reboiler duty penalties and solvent degradation over time.

## Physics Added Over F1a
1. **Part-load reboiler duty:** Off-design L/G ratio increases specific duty. q(PLR) = q_design * (1.3 - 0.3*PLR).
2. **Solvent degradation:** Capacity loss = 0.02 * operating_hours/1000 (2% per 1000h). Degraded solvent needs more circulation.
3. **Electrical consumption:** Fan power scales inversely with PLR; pump power scales with 1/capacity.
4. **Total energy penalty:** Expressed as % of reference plant output.

## Inputs
| Parameter | Unit | Range | Default |
|-----------|------|-------|---------|
| flue_gas_flow_mol_s | mol/s | 10-1000 | 100 |
| co2_concentration | mol/mol | 0.04-0.15 | 0.12 |
| capture_rate | - | 0.5-0.95 | 0.90 |
| PLR | - | 0.3-1.0 | 1.0 |
| operating_hours | hours | 0-100000 | 0 |

## Outputs
| Parameter | Unit |
|-----------|------|
| co2_captured_kg_h | kg/h |
| reboiler_duty_gj_ton | GJ/tCO2 |
| electrical_kwh_ton | kWh/tCO2 |
| solvent_degradation_pct | % |
| total_energy_penalty_pct | % |

## Key Parameters
- reboiler_duty_design = 3.5 GJ/tCO2
- electrical_design = 40 kWh/tCO2
- L_G_design = 2.5 mol/mol
- solvent_capacity_initial = 0.5 mol_CO2/mol_amine
- degradation_rate = 2%/1000h

## References
- Abu-Zahra, M.R.M. et al. (2007). Int. J. Greenhouse Gas Control, 1(1), 37-46.
- Rochelle, G.T. (2009). Science, 325(5948), 1652-1654.
