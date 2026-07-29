# EC074 -- Plate Heat Exchanger -- F2a Lumped Dynamic Model

## Model Description
Two-fluid lumped capacitance model with thermal mass and transient response ODE.
Uses epsilon-NTU method for UA estimation with flow-dependent scaling.

## Governing Equations
```
dT_h/dt = (m_dot_h*cp_h*(T_h_in - T_h) - UA*(T_h - T_c) - UA_loss*(T_h - T_amb)) / (M_h*cp_h)
dT_c/dt = (m_dot_c*cp_c*(T_c_in - T_c) + UA*(T_h - T_c) - UA_loss*(T_c - T_amb)) / (M_c*cp_c)
```

## Inputs
| Parameter | Unit | Default | Description |
|-----------|------|---------|-------------|
| m_dot_hot | kg/s | 1.2 | Hot-side mass flow rate |
| m_dot_cold | kg/s | 1.0 | Cold-side mass flow rate |
| T_hot_in | K | 353.15 | Hot inlet temperature |
| T_cold_in | K | 293.15 | Cold inlet temperature |
| T_hot_init | K | 293.15 | Initial hot-side temperature |
| T_cold_init | K | 293.15 | Initial cold-side temperature |

## Outputs
| Parameter | Unit | Description |
|-----------|------|-------------|
| T_hot_out | K | Hot-side outlet temperature |
| T_cold_out | K | Cold-side outlet temperature |
| Q_transfer | W | Heat transfer rate |
| effectiveness | - | HX effectiveness |
| UA | W/K | Effective UA value |

## References
- Incropera & DeWitt (2011), Fundamentals of Heat and Mass Transfer
- Shah & Sekulic (2003), Fundamentals of Heat Exchanger Design
