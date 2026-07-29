# EC028 -- Lead-Acid Battery -- F2a: 1-RC Equivalent Circuit Model

## Model Description

1-RC Equivalent Circuit Model for a 12V/100Ah flooded lead-acid battery (6 cells in series). Captures instantaneous resistive drop and transient polarization dynamics. Lead-acid batteries have longer RC time constants (~60s) compared to Li-ion due to slower electrochemical kinetics.

### Governing Equations

```
V_terminal = OCV(SOC) - I * R0(SOC) - V_rc
dV_rc/dt   = I / C1(SOC) - V_rc / (R1(SOC) * C1(SOC))
dSOC/dt    = -I / (Q_nom * 3600)
```

### Parameters (Nominal at SOC = 0.5)

| Parameter | Value | Unit | Description |
|-----------|-------|------|-------------|
| Q_nom | 100 | Ah | Nominal capacity |
| R0 | 0.005 | Ohm | Series resistance |
| R1 | 0.003 | Ohm | Polarization resistance |
| C1 | 20000 | F | Polarization capacitance |
| tau1 | 60 | s | RC time constant |
| V_max | 14.4 | V | Maximum voltage |
| V_min | 10.5 | V | Cutoff voltage |

## Data Source

Copetti et al. (1993), J. Power Sources; Manwell & McGowan (1993), Sol. Energy.

## Limitations

- Isothermal model (no temperature dependence, important for lead-acid)
- No Peukert effect (capacity depends on discharge rate)
- Single RC pair; lead-acid has complex double-layer effects
- No sulfation or stratification aging
