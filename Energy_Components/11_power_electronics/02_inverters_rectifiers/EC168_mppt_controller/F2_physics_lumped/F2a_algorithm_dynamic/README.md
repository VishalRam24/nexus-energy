# EC168 -- MPPT Controller -- F2a Algorithm Dynamic

## Model Description
Physics-lumped ODE model of a Maximum Power Point Tracking (MPPT) controller using Perturb & Observe (P&O) algorithm coupled with buck converter averaged dynamics.

## Physics
- **PV model:** Single-diode with series and shunt resistance (Newton-Raphson solver)
- **MPPT algorithm:** Perturb & Observe with configurable step size and sampling period
- **Buck converter:** Averaged model with L, C, parasitic resistance
- **PI controller:** Tracks MPPT voltage reference via duty cycle adjustment

## Inputs
| Parameter | Unit | Default | Range |
|-----------|------|---------|-------|
| irradiance | W/m2 | 1000 | 0-1200 |
| T_cell | K | 298.15 | 273-348 |
| dt | s | 0.001 | - |
| duration_s | s | 1.0 | 0.01-60 |

## Outputs
| Variable | Unit | Description |
|----------|------|-------------|
| V_pv | V | PV terminal voltage |
| I_pv | A | PV output current |
| P_pv | W | PV power |
| V_ref | V | MPPT voltage reference |
| duty_cycle | - | Buck converter duty cycle |
| I_L | A | Inductor current |
| V_out | V | Output voltage |
| P_out | W | Output power to load |
| tracking_efficiency | - | P_pv / P_mpp |

## References
- Esram & Chapman (2007), IEEE Trans. Energy Conv., 22(2), 439-449
- Femia et al. (2005), IEEE Trans. Power Electron., 20(4), 963-973

## Limitations
- Averaged buck converter model (no switching ripple)
- Single PV panel (no string/array mismatch)
- P&O only (no incremental conductance or advanced algorithms)
