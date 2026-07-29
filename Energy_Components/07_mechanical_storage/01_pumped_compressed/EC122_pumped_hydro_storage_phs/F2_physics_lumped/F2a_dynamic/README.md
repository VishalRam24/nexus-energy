# EC122 -- Pumped Hydro Storage (PHS) -- F2a Dynamic ODE Model

## Model Description
Physics-lumped dynamic model for reversible pump-turbine pumped hydro storage.
Couples waterway inertia (penstock momentum equation), rotor dynamics, and reservoir
level tracking through an ODE system solved with `scipy.integrate.solve_ivp`.

## Physics
- **Waterway momentum**: `rho*L/A * dQ/dt = rho*g*H_eff - runner_forces`
- **Rotor dynamics**: `J * d_omega/dt = T_hydraulic - T_electrical - T_friction`
- **Reservoir balance**: `dH_up/dt = -Q/A_upper`, `dH_low/dt = Q/A_lower`
- **Water hammer**: Simplified Joukowsky equation for pressure transients
- **Efficiency maps**: Hill chart approximation for turbine and pump modes

## Inputs
| Parameter | Unit | Description |
|-----------|------|-------------|
| P_electrical_W | W | Electrical power demand (positive=generate, negative=pump) |
| mode | str | 'turbine', 'pump', or 'idle' |
| Q0 | m3/s | Initial flow rate |
| omega0 | rad/s | Initial angular speed |
| H_up0, H_low0 | m | Initial reservoir levels |
| dt | s | Output time step |
| duration_s | s | Simulation duration |

## Outputs
| Variable | Unit | Description |
|----------|------|-------------|
| t | s | Time array |
| Q | m3/s | Flow rate |
| omega | rad/s | Rotor speed |
| H_upper, H_lower | m | Reservoir levels |
| H_net | m | Net head |
| P_hydraulic | W | Hydraulic power |
| efficiency | - | Operating efficiency |
| SOC | - | State of charge |
| E_stored | J | Stored energy |

## References
- Nicolet (2007), Hydroacoustic Modelling, EPFL
- Chaudhry (2014), Applied Hydraulic Transients, Springer

## Limitations
- Simplified efficiency maps (not full hill chart from CFD)
- 0D lumped penstock (no distributed water hammer wave propagation)
- Single penstock, single turbine unit
