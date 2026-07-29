# EC009 -- AEL -- F2a Electrochemical Model

## Model Description
Physics-lumped electrochemical model for alkaline water electrolysis with bubble coverage.

**Cell voltage:** `V_cell = E_rev(T) + eta_act_anode + eta_act_cathode + eta_ohm`
- Activation: Butler-Volmer (separate anode OER and cathode HER)
- Ohmic: KOH conductivity + diaphragm + Bruggeman bubble correction
- Bubble coverage: `theta = k_bubble * j^0.3`

## Inputs
| Parameter | Unit | Default | Range |
|-----------|------|---------|-------|
| current_density | A/m2 | - | [100, 6000] |
| T_K | K | 353 | [323, 363] |
| koh_wt_pct | % | 30 | [20, 40] |

## Outputs
| Parameter | Unit |
|-----------|------|
| voltage | V |
| h2_production | mol/s |
| efficiency | dimensionless |
| bubble_coverage | dimensionless |

## References
- Ulleberg (2003), Int. J. Hydrogen Energy, 28(1), 21-33.
- Haug et al. (2017), Int. J. Hydrogen Energy, 42(25), 15689-15707.
