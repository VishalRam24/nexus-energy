# EC036 -- VRFB -- F2a Stack Model

## Model Description
Physics-lumped dynamic model of a Vanadium Redox Flow Battery (VRFB) stack with coupled tank SOC dynamics.

**Cell voltage:** `E_cell = E_nernst(SOC) - eta_act - eta_ohm - eta_conc`
- Nernst: `E0 + 2*(R*T)/(n*F) * ln(SOC/(1-SOC))`
- Activation: Butler-Volmer symmetric form
- Ohmic: Area-specific resistance
- Concentration: Flow-rate-dependent limiting current

**Tank dynamics:** `dSOC/dt = -I / (n*F*c_total*V_tank)`

**Pump hydraulics:** `P_pump = 2 * delta_P * Q / eta_pump`

## Inputs
| Parameter | Unit | Default | Range |
|-----------|------|---------|-------|
| current_A | A | - | [-120, 120] |
| flow_rate_L_min | L/min | 10 | [1, 30] |
| dt | s | 1.0 | - |
| duration_s | s | 3600 | - |
| soc_init | - | 0.5 | [0.1, 0.9] |

## Outputs
| Parameter | Unit |
|-----------|------|
| voltage | V |
| soc | dimensionless |
| power_stack | W |
| power_pump | W |
| net_power | W |
| efficiency | dimensionless |

## References
- Blanc, C., Rufer, A. (2010). Multiphysics and Energetic Modeling of a VRFB.
- Shah, A.A. et al. (2011). Electrochimica Acta, 56(3), 1570-1578.
