# EC078 — Hot Water Tank TES — F1b Stratified

## Model

Multi-node (N=10) vertical stratification model. Each node energy balance:

```
m_node * cp * dT_i/dt = Q_advection_i - UA_node*(T_i - T_amb) + k_mix*(T_{i-1} + T_{i+1} - 2*T_i)
```

Features:
- Charging: hot water enters top, cold exits bottom
- Discharging: cold water enters bottom, hot drawn from top
- Inter-node conductive/turbulent mixing
- Buoyancy correction (no temperature inversions)

## Inputs

| Parameter | Unit | Range | Default |
|-----------|------|-------|---------|
| T_inlet_hot | degC | 40-95 | 80 |
| T_inlet_cold | degC | 5-30 | 15 |
| flow_rate_charge | kg/s | 0-2 | 0 |
| flow_rate_discharge | kg/s | 0-2 | 0 |
| T_ambient | degC | -10-40 | 20 |
| duration_s | s | 1-86400 | 3600 |

## Outputs

| Parameter | Unit |
|-----------|------|
| T_nodes | degC (array of 10) |
| T_outlet_hot | degC |
| T_outlet_cold | degC |
| stored_energy_kwh | kWh |
| stratification_efficiency | - |

## References

- Duffie & Beckman (2013), ch. 8
- TRNSYS Type 60 (Newton, 1995)
- De Cesaro Oliveski et al. (2003), Applied Thermal Engineering
