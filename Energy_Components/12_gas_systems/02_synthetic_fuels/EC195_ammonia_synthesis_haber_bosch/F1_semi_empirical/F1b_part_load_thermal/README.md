# EC195 — Ammonia Synthesis (Haber-Bosch) — F1b Part-Load + Thermal Model

## Overview
Extends the F1a per-pass conversion model with part-load effects, recycle loop dynamics, compression energy, and purge fraction.

**Reaction:** N2 + 3H2 -> 2NH3 (DH = -92 kJ/mol N2)

## Physics Added Over F1a
1. **Part-load pressure effect:** P_eff = P_design * (0.85 + 0.15*PLR). Off-design compressor operation.
2. **Recycle ratio:** R = 1/X_sp - 1. Increases dramatically at low conversion.
3. **Multi-stage compression energy:** Isentropic compression with intercooling, 3 stages, eta=0.85.
4. **Purge fraction:** Increases at part-load to control inert buildup.
5. **Specific energy (kWh/ton):** Accounts for compression + heating + recycle penalties.

## Inputs
| Parameter | Unit | Range | Default |
|-----------|------|-------|---------|
| n2_flow_mol_s | mol/s | >0 | 1.0 |
| h2_n2_ratio | mol/mol | 2.5-3.5 | 3.0 |
| PLR | - | 0.3-1.0 | 1.0 |
| pressure_bar | bar | 100-300 | 200 |
| temperature_c | degC | 350-550 | 450 |

## Outputs
| Parameter | Unit |
|-----------|------|
| nh3_production_mol_s | mol/s |
| single_pass_conversion | - |
| recycle_ratio | - |
| energy_kwh_per_ton | kWh/ton NH3 |
| purge_fraction | - |

## Key Parameters
- P_design = 200 bar, T_design = 450 degC
- conversion_design = 0.15 (single-pass)
- compression_stages = 3, eta_compressor = 0.85
- purge_fraction_design = 0.05

## References
- Appl, M. (2011). Ammonia. In Ullmann's Encyclopedia of Industrial Chemistry.
- Patil, A. et al. (2015). Procedia Engineering, 138, 229-236.
