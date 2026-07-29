# EC098 -- Organic Rankine Cycle (ORC) -- F2a Thermodynamic Cycle Steady-State

## Model Description
Physics-lumped Rankine cycle model with R245fa working fluid. Four state points (pump, evaporator, expander, condenser) with polynomial property correlations (no CoolProp dependency). Includes part-load with expander/pump efficiency corrections and thermal inertia dynamics.

## Fidelity
F2a -- Thermodynamic cycle steady-state with part-load and dynamic thermal mass.

## Inputs
| Parameter | Unit | Range | Description |
|-----------|------|-------|-------------|
| P_evap | Pa | 500k-3M | Evaporator pressure |
| P_cond | Pa | 100k-500k | Condenser pressure |
| load_fraction | - | 0.1-1.0 | Part-load fraction |
| superheat | K | 0-30 | Superheating at expander inlet |
| mode | - | steady/dynamic | Simulation mode |

## Outputs
| Parameter | Unit | Description |
|-----------|------|-------------|
| W_net | W | Net power output |
| eta_thermal | - | Thermal efficiency |
| Q_in | W | Heat input |
| Q_out | W | Heat rejection |
| m_dot | kg/s | Working fluid mass flow rate |
| state_points | dict | T, P, h, s at each state point |

## Working Fluid
R245fa (1,1,1,3,3-pentafluoropropane) with polynomial correlations fitted from NIST data. No external property library required.

## References
- Quoilin et al. (2013), Renewable and Sustainable Energy Reviews, 22, 168-186
- Lemort et al. (2009), Applied Thermal Engineering, 29, 1684-1694
- NIST WebBook for R245fa property data

## Limitations
- Polynomial property fits valid for 280-420 K range
- Single-component working fluid only
- No recuperator modeled (simple cycle)
- Part-load model is empirical correction, not detailed off-design
