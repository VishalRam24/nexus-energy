# EC031 -- Sodium-Ion Battery -- F2a: 1-RC Equivalent Circuit Model

## Model Description

1-RC Equivalent Circuit Model for a sodium-ion battery (layered oxide cathode / hard carbon anode). Na-ion batteries have higher internal resistance than Li-ion due to the larger Na+ ionic radius, but offer cost advantages and excellent low-temperature performance.

### Governing Equations

```
V_terminal = OCV(SOC) - I * R0(SOC) - V_rc
dV_rc/dt   = I / C1(SOC) - V_rc / (R1(SOC) * C1(SOC))
dSOC/dt    = -I / (Q_nom * 3600)
```

### Parameters (Nominal at SOC = 0.5)

| Parameter | Value | Unit | Description |
|-----------|-------|------|-------------|
| Q_nom | 3.0 | Ah | Nominal capacity |
| R0 | 0.020 | Ohm | Series resistance |
| R1 | 0.012 | Ohm | Polarization resistance |
| C1 | 2500 | F | Polarization capacitance |
| tau1 | 30 | s | RC time constant |
| V_max | 3.9 | V | Maximum voltage |
| V_min | 1.5 | V | Cutoff voltage |

## Data Source

CATL first-gen Na-ion press release data; Tremblay & Dessaint (2009) framework adapted.

## Limitations

- Isothermal model (no temperature dependence)
- Single RC pair captures ~30s dynamics only
- No degradation/aging
- Na-ion parameter data is limited; values are estimates based on early publications
