# EC018 -- LFP Battery -- F2a: 1-RC Equivalent Circuit Model

## Model Description

1-RC Equivalent Circuit Model for LFP (LiFePO4) lithium-ion battery. Captures instantaneous resistive drop and transient polarization dynamics.

### Governing Equations

```
V_terminal = OCV(SOC) - I * R0(SOC) - V_rc
dV_rc/dt   = I / C1(SOC) - V_rc / (R1(SOC) * C1(SOC))
dSOC/dt    = -I / (Q_nom * 3600)
```

### Parameters (Nominal at SOC = 0.5)

| Parameter | Value | Unit | Description |
|-----------|-------|------|-------------|
| Q_nom | 3.3 | Ah | Nominal capacity |
| R0 | 0.012 | Ohm | Series resistance |
| R1 | 0.008 | Ohm | Polarization resistance |
| C1 | 4000 | F | Polarization capacitance |
| tau1 | 32 | s | RC time constant |
| V_max | 3.6 | V | Maximum voltage |
| V_min | 2.0 | V | Cutoff voltage |

## Data Source

Chen et al. (2020). J. Electrochem. Soc., 167, 080534; A123 ANR26650M1B datasheet.

## Limitations

- Isothermal model (no temperature dependence)
- Single RC pair captures ~32s dynamics only
- No degradation/aging
- LFP flat voltage plateau makes SOC estimation challenging
