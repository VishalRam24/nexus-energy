# EC079 — Molten Salt Thermal Energy Storage — F1a Fully Mixed

## Overview
0D fully mixed energy balance model for two-tank solar salt (60% NaNO₃ + 40% KNO₃) thermal
energy storage systems, as used in concentrating solar power (CSP) plants.

## Model Card

| Property | Value |
|---|---|
| EC ID | EC079 |
| Fidelity | F1a |
| Fluid | Solar salt: 60% NaNO₃ + 40% KNO₃ |
| Volume | 1000 m³ |
| Mass | 1,800,000 kg (ρ = 1800 kg/m³) |
| cp | 1500 J/(kg·K) |
| T_hot | 565 °C |
| T_cold | 290 °C |
| Energy capacity | 206.25 MWh |
| UA_loss | 50 W/K |
| T_solidification | 220 °C (must stay above) |

## Inputs / Outputs

| Input | Unit | Range | Description |
|---|---|---|---|
| temperature | °C | 220 – 580 | Current bulk salt temperature |
| q_charge | kW | 0 – 100,000 | Thermal power input |
| q_discharge | kW | 0 – 100,000 | Thermal power output |
| t_ambient | °C | -20 – 50 | Ambient temperature (default 25°C) |

| Output | Unit | Description |
|---|---|---|
| dT_dt | K/s | Rate of temperature change |
| energy_stored_mwh | MWh | Energy stored above T_cold |
| soc | - | State of charge [0, 1] |
| heat_loss_kw | kW | Heat loss to environment |

## Physics

```
dT/dt = (Q_charge - Q_discharge - Q_loss) / (m × cp)
Q_loss = UA × (T - T_ambient)   [W]

E_stored = m × cp × (T - T_cold)   [J]
SOC = (T - T_cold) / (T_hot - T_cold)
```

## Energy Capacity Calculation

```
E = 1,800,000 kg × 1500 J/(kg·K) × (565 - 290) K
  = 1,800,000 × 1500 × 275
  = 7.425 × 10¹¹ J
  = 206.25 MWh
```

This is a 206 MWh system — consistent with a ~50 MWe CSP plant operating at 4 hours full load.

## Tests (14/14 passing)
- Output key completeness, EC ID, fidelity
- Energy ≥ 0 everywhere
- SOC = 0 at T_cold, SOC = 1 at T_hot
- SOC always in [0, 1]
- Heat loss > 0 when T > T_ambient
- dT/dt > 0 when charging
- dT/dt < 0 when discharging
- dT/dt < 0 when idle (heat loss)
- Energy capacity ~206 MWh
- Energy monotonically increases with temperature
- Simulation roundtrip (charge → SOC increases)
- Benchmark: 1000 predictions < 1 second

## Data Sources
- Herrmann, U., Kelly, B., & Price, H. (2004). "Two-tank molten salt storage for parabolic trough solar power plants." _Energy_, 29(5–6), 883–893.
- Kearney, D. et al. (2003). "Assessment of a molten salt heat transfer fluid in a parabolic trough solar field." _J. Solar Energy Eng._, 125, 170–176.

## Known Limitations
- Fully mixed (0D): no temperature stratification
- Constant properties (cp, ρ assumed temperature-independent)
- No freeze protection logic or solidification dynamics below 220°C
- Two-tank vs single-tank configurations not distinguished
