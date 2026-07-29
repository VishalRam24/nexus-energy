# EC019 -- NMC Battery -- F2a: 1-RC Equivalent Circuit Model

## Model Description

1-RC Equivalent Circuit Model (ECM) for NMC811 lithium-ion battery. This is a dynamic ODE-based model that captures both the instantaneous resistive voltage drop and the transient polarization dynamics through an RC parallel network.

### Governing Equations

```
V_terminal = OCV(SOC) - I * R0(SOC) - V_rc
dV_rc/dt   = I / C1(SOC) - V_rc / (R1(SOC) * C1(SOC))
dSOC/dt    = -I / (Q_nom * 3600)
```

### Parameters (Nominal at SOC = 0.5)

| Parameter | Value | Unit | Description |
|-----------|-------|------|-------------|
| Q_nom | 5.0 | Ah | Nominal capacity |
| R0 | 0.015 | Ohm | Series resistance |
| R1 | 0.010 | Ohm | Polarization resistance |
| C1 | 3000 | F | Polarization capacitance |
| tau1 | 30 | s | RC time constant (R1 * C1) |
| V_max | 4.2 | V | Maximum voltage |
| V_min | 2.5 | V | Cutoff voltage |

All resistance and capacitance parameters are SOC-dependent via quadratic multipliers.

## Inputs

| Name | Unit | Range | Description |
|------|------|-------|-------------|
| current | A | [-25, 25] | Current (positive = discharge) |
| dt | s | > 0 | Time step |
| soc_init | - | [0, 1] | Initial state of charge |

## Outputs

| Name | Unit | Description |
|------|------|-------------|
| voltage | V | Terminal voltage time series |
| soc | - | State of charge time series |
| v_rc | V | RC polarization voltage |
| power | W | Electrical power (V * I) |
| time | s | Time array |

## Data Source

Chen et al. (2020). "Development of Experimental Techniques for Parameterization of Multi-scale Lithium-ion Battery Models." J. Electrochem. Soc., 167, 080534.

## Limitations

- Isothermal (no temperature dependence) -- see F2b for thermal coupling
- Single RC pair (captures ~30s dynamics, misses slower diffusion)
- No degradation/aging -- see F2c
- SOC-dependent parameters use simplified quadratic fits
