# EC128 — Conventional Hydroelectric Dam — F2a Dynamic ODE Model

## Fidelity
**F2a** — Physics-lumped dynamic model with penstock momentum, turbine-governor, and reservoir dynamics.

## Model Description
Three coupled ODEs:
- **Reservoir level**: dH/dt = (Q_inflow - Q_turbine) / A_reservoir
- **Penstock flow**: dQ/dt = (g·A/L)·(H_net - h_friction) + governor blend
- **Governor gate**: dG/dt = (G_ref - G) / τ_gov with rate limiting

Francis turbine with parabolic efficiency map. Darcy-Weisbach penstock friction.

## Inputs

| Parameter | Unit | Default | Range |
|-----------|------|---------|-------|
| G_ref | - | 0.5 | [0.05, 1.0] |
| Q_inflow | m³/s | 100 | [0, 500] |
| H0 | m | 80 | [10, 150] |
| dt | s | 1.0 | - |
| duration_s | s | 3600 | - |

## Outputs

| Variable | Unit | Description |
|----------|------|-------------|
| H_reservoir | m | Reservoir water level |
| Q_penstock | m³/s | Penstock volumetric flow |
| Q_turbine | m³/s | Turbine flow from gate equation |
| G_gate | - | Gate opening fraction |
| P_output | W | Electrical power output |
| efficiency | - | Francis turbine efficiency |
| Q_inflow | m³/s | Natural inflow |

## Key Parameters
- Rated: 200 MW, 80 m head, 300 m³/s rated flow
- Francis turbine peak efficiency: 0.93
- Governor time constant: 5 s
- Gate rate limit: 0.1 per second

## References
- Kundur (1994), Power System Stability and Control, McGraw-Hill
- IEEE Std 1207 — Hydroelectric Generating Stations
- USBR Engineering Monograph No. 20

## Limitations
- Lumped parameter (no distributed penstock wave propagation)
- Simplified efficiency map (no full hill chart)
- No cavitation or draft tube modeling
- Synchronous generator at fixed speed (no speed dynamics)
