# EC078 — Sensible Heat Storage (Hot Water Tank) — F1a Fully Mixed

## Model Overview

Lumped-parameter perfectly stirred tank model. Temperature evolves according to:

```
dT/dt = (Q_charge - Q_discharge - UA*(T - T_amb)) / (m * cp)

energy_stored = m * cp * (T - T_min)   [J -> kWh]
soc           = (T - T_min) / (T_max - T_min)
heat_loss     = UA * (T - T_amb)       [W]
```

## Parameters

| Parameter | Value | Unit | Note |
|-----------|-------|------|------|
| volume | 500 | L | Tank volume |
| UA_loss | 3 | W/K | Overall heat loss conductance |
| T_min | 30 | degC | SOC=0 reference |
| T_max | 90 | degC | SOC=1 upper limit |
| T_set | 60 | degC | Design set-point |
| cp_water | 4186 | J/(kg K) | |

## Inputs / Outputs

**Inputs:**

| Name | Unit | Range | Default |
|------|------|-------|---------|
| temperature | degC | 0 – 100 | required |
| q_charge | W | 0 – 50000 | 0 |
| q_discharge | W | 0 – 50000 | 0 |
| t_ambient | degC | –10 to 40 | 20 |

**Outputs:**

| Name | Unit |
|------|------|
| dT_dt | K/s |
| energy_stored_kwh | kWh |
| soc | dimensionless |
| heat_loss_w | W |

## Usage

```python
from scripts.predict import ComponentModel
model = ComponentModel()
result = model.predict({"temperature": 60.0, "q_charge": 5000.0, "q_discharge": 3000.0})
print(result)
```

## Reference

Duffie, J.A., Beckman, W.A. (2013). *Solar Engineering of Thermal Processes*, 4th ed.
John Wiley & Sons, ch. 8.

## Limitations

- Perfectly mixed assumption ignores temperature stratification (F1b adds multi-node stratification).
- Fixed UA does not account for insulation variation with temperature.
- Inlet/outlet flow mixing effects are not modeled.
