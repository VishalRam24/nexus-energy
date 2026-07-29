# EC193 — Methanation Reactor — F1b Part-Load + Thermal Model

## Overview
Extends the F1a equilibrium model with part-load ratio (PLR) effects and exothermic heat recovery.

**Reaction:** CO2 + 4H2 -> CH4 + 2H2O (DH = -165 kJ/mol)

## Physics Added Over F1a
1. **Part-load conversion drop:** PLR_factor = a0 + a1*PLR + a2*PLR^2. At part-load, reduced feed lowers catalyst bed temperature and conversion.
2. **Reactor temperature correction:** T_eff = T_reactor - (1-PLR)*30 degC.
3. **Exothermic heat recovery:** Q = X * |DH_rxn| * n_CO2 * PLR * f_recovery [kW].
4. **Selectivity degradation:** S = S_design * (0.8 + 0.2*PLR).
5. **Overall efficiency:** Includes CH4 chemical energy + heat recovery credit vs H2 input.

## Inputs
| Parameter | Unit | Range | Default |
|-----------|------|-------|---------|
| co2_flow_mol_s | mol/s | >0 | 1.0 |
| h2_co2_ratio | mol/mol | 3.5-5.0 | 4.0 |
| PLR | - | 0.3-1.0 | 1.0 |
| T_reactor_degC | degC | 200-500 | 300 |
| pressure_bar | bar | 1-30 | 10 |

## Outputs
| Parameter | Unit |
|-----------|------|
| ch4_production_mol_s | mol/s |
| conversion | - |
| heat_recovery_kw | kW |
| overall_efficiency | - |
| selectivity | - |

## Key Parameters
- catalyst_mass = 100 kg (Ni/Al2O3)
- GHSV = 5000/h
- f_recovery = 0.85
- DH_reaction = -165 kJ/mol
- PLR_coeffs = [0.2, 1.2, -0.4]

## References
- Gao, J. et al. (2012). RSC Advances, 2, 2358-2368.
- Gotz, M. et al. (2016). Renewable Energy, 85, 1371-1390.
