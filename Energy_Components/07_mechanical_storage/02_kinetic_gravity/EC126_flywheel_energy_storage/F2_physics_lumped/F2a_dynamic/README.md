# EC126 -- Flywheel Energy Storage -- F2a Dynamic ODE Model

## Model Description
Physics-lumped dynamic model for flywheel energy storage systems. Solves the
rotational dynamics ODE with motor/generator torque, friction losses (windage
and bearing), and SOC tracking.

## Physics
- **Rotational dynamics**: `J * d_omega/dt = T_motor - T_load - T_friction(omega)`
- **Stored energy**: `E = 0.5 * J * omega^2`
- **Friction**: `T_friction = c_windage * omega^2 + T_bearing`
- **Motor/generator efficiency**: Maps electrical power to shaft torque

## Inputs
| Parameter | Unit | Description |
|-----------|------|-------------|
| P_command_W | W | Power command (positive=charge, negative=discharge) |
| omega0 | rad/s | Initial angular speed |
| dt | s | Output time step |
| duration_s | s | Simulation duration |

## Outputs
| Variable | Unit | Description |
|----------|------|-------------|
| t | s | Time array |
| omega | rad/s | Angular speed |
| E_stored | J | Stored kinetic energy |
| SOC | - | State of charge |
| P_loss | W | Friction power loss |
| efficiency | - | Instantaneous efficiency |

## References
- Amiryar & Pullen (2017), Applied Sciences 7(3):286
- Beacon Power flywheel specifications

## Limitations
- Simplified motor/generator efficiency (constant, not speed-dependent map)
- No thermal model for bearings or motor
- Single rotor (no multi-flywheel array dynamics)
