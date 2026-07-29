# EC020 -- NCA Battery -- F2a: 1-RC Equivalent Circuit Model

## Model Description

1-RC Equivalent Circuit Model for NCA (LiNiCoAlO2) lithium-ion battery. Captures instantaneous resistive drop and transient polarization dynamics.

### Governing Equations

```
V_terminal = OCV(SOC) - I * R0(SOC) - V_rc
dV_rc/dt   = I / C1(SOC) - V_rc / (R1(SOC) * C1(SOC))
dSOC/dt    = -I / (Q_nom * 3600)
```

### Parameters (Nominal at SOC = 0.5)

| Parameter | Value | Unit | Description |
|-----------|-------|------|-------------|
| Q_nom | 3.5 | Ah | Nominal capacity |
| R0 | 0.014 | Ohm | Series resistance |
| R1 | 0.009 | Ohm | Polarization resistance |
| C1 | 3500 | F | Polarization capacitance |
| tau1 | 31.5 | s | RC time constant |
| V_max | 4.2 | V | Maximum voltage |
| V_min | 2.5 | V | Cutoff voltage |

## Data Source

Tremblay & Dessaint (2009), IEEE Trans. Veh. Technol.; Panasonic NCR18650B datasheet.

## Limitations

- Isothermal model (no temperature dependence)
- Single RC pair captures ~31.5s dynamics only
- No degradation/aging
