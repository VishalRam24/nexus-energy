# EC078 -- Hot Water Tank TES -- F2a 1D Transient Model

## Model Description
20-node 1D vertically stratified hot water tank with transient ODE system.
Includes axial conduction, charge/discharge advection, heat loss to ambient,
and buoyancy-driven mixing to prevent temperature inversions.

## Governing Equations
For each node i (0=top, 19=bottom):
```
M_i*cp * dT_i/dt = Q_conduction + Q_charge + Q_discharge - Q_loss + Q_mixing
```

## Inputs
| Parameter | Unit | Default | Description |
|-----------|------|---------|-------------|
| m_dot_charge | kg/s | 0.5 | Charge flow rate |
| T_charge_in | K | 353.15 | Hot charge inlet temperature |
| m_dot_discharge | kg/s | 0.0 | Discharge flow rate |
| T_discharge_in | K | 288.15 | Cold return temperature |

## Outputs
| Parameter | Unit | Description |
|-----------|------|-------------|
| T_profiles | K | Full 20-node temperature profile over time |
| T_top / T_bottom | K | Top and bottom node temperatures |
| E_stored_kWh | kWh | Total stored energy above ambient |
| stratification_K | K | Temperature difference top-bottom |

## References
- Kleinbach et al. (1993), Solar Energy, 50(2), 155-166
- Duffie & Beckman (2013), Solar Engineering of Thermal Processes
