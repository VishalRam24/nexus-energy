# EC010 -- SOEC -- F2a Electrochemical Model

## Model Description
Physics-lumped electrochemical model for solid oxide electrolysis at high temperature (700-900 degC).

**Cell voltage:** `V_cell = E_nernst(T, U) + eta_act + eta_ohm`
- Nernst with steam utilization effect
- Activation: Butler-Volmer (Arrhenius T-dependent j0)
- Ohmic: YSZ ionic conductivity (Arrhenius)
- Thermal mode detection: endothermic (V < V_tn) vs exothermic

## Inputs
| Parameter | Unit | Default | Range |
|-----------|------|---------|-------|
| current_density | A/cm2 | - | [0.01, 2.0] |
| T_K | K | 1073.15 | [923, 1173] |
| steam_utilization | - | 0.5 | [0.1, 0.9] |

## Outputs
| Parameter | Unit |
|-----------|------|
| voltage | V |
| h2_production | mol/s |
| efficiency | dimensionless |
| thermal_mode | 1=endothermic, -1=exothermic |

## References
- Ni et al. (2007), Chem. Eng. Tech., 29(6), 636-642.
- Udagawa et al. (2007), J. Power Sources, 166(1), 127-136.
