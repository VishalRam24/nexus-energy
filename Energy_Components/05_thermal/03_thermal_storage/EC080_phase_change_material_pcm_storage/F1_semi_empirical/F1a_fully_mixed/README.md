# EC080 — PCM Thermal Energy Storage — F1a Latent Heat Model

## Overview
Three-region latent heat model for paraffin RT42 phase-change material (PCM) thermal energy
storage. Captures solid, mushy (phase change), and liquid regions with correct energy absorption
physics: during phase change, all net heat goes to melting/solidifying (dT/dt = 0, df/dt ≠ 0).

## Model Card

| Property | Value |
|---|---|
| EC ID | EC080 |
| Fidelity | F1a |
| PCM Material | Paraffin RT42 |
| Mass | 500 kg |
| T_melt | 42 °C |
| Latent heat (L) | 174 kJ/kg |
| cp_solid | 2.0 kJ/(kg·K) |
| cp_liquid | 2.0 kJ/(kg·K) |
| UA_loss | 5 W/K |
| Full energy capacity | ~38 kWh |

## Inputs / Outputs

| Input | Unit | Range | Description |
|---|---|---|---|
| temperature | °C | 0 – 80 | Current PCM bulk temperature |
| liquid_fraction | - | 0 – 1 | Current liquid fraction (0=solid, 1=liquid) |
| q_charge | W | 0 – 10,000 | Thermal input power |
| q_discharge | W | 0 – 10,000 | Thermal output power |

| Output | Unit | Description |
|---|---|---|
| dT_dt | K/s | Temperature rate (0 in mushy zone) |
| d_fraction_dt | 1/s | Liquid fraction rate (0 outside mushy zone) |
| energy_stored_kwh | kWh | Total energy stored |
| soc | - | State of charge [0, 1] |

## Physics — Three-Region Model

### Solid region (T < T_solidus = 40°C)
```
dT/dt = Q_net / (m × cp_solid)
df/dt = 0
```

### Mushy region (T = Tm = 42°C, 0 ≤ f ≤ 1)
```
dT/dt = 0         (temperature pinned at T_melt)
df/dt = Q_net / (m × L)
```

### Liquid region (T > T_liquidus = 44°C)
```
dT/dt = Q_net / (m × cp_liquid)
df/dt = 0
```

### Heat loss
```
Q_loss = UA × (T - T_ambient)   [W]
Q_net = Q_charge - Q_discharge - Q_loss
```

### Total energy stored
```
E = m × [cp_s × (Tm - T_ref) + f × L + cp_l × max(0, T - Tm)]
```

## Energy Budget

| Component | Energy |
|---|---|
| Sensible (solid, 0→42°C) | 500 × 2.0 × 42 / 3.6 = 11.67 kWh |
| Latent (full melting) | 500 × 174 / 3600 = 24.17 kWh |
| Liquid buffer (42→50°C) | 500 × 2.0 × 8 / 3.6 = 2.22 kWh |
| **Total** | **~38 kWh** |

## Tests (14/14 passing)
- Output key completeness, EC ID, fidelity
- SOC and liquid fraction in [0, 1]
- Energy increases with liquid fraction (latent heat)
- dT/dt = 0 in mushy zone
- df/dt > 0 when charging in mushy zone (melting)
- df/dt < 0 when discharging in mushy zone (solidifying)
- dT/dt > 0 in solid and liquid when charging
- Latent heat magnitude ~24.2 kWh
- Simulation: energy increases monotonically during charge-only
- Benchmark: 1000 predictions < 1 second

## Data Sources
- Mehling, H. & Cabeza, L.F. (2008). _Heat and Cold Storage with PCM_. Springer, Berlin.
- Rubitherm Technologies GmbH — RT42 Product Datasheet. Available: https://www.rubitherm.eu

## Known Limitations
- Fully mixed (0D): no spatial temperature gradients within PCM
- Sharp mushy zone (±2°C); real PCMs have broader melting ranges
- Constant cp and thermal conductivity; actual values vary with phase and temperature
- No subcooling or superheating effects
- No convection enhancement during liquid phase
