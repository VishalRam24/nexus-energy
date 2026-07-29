# EC048 — Perovskite Solar Cell — F1b Two-Diode + Hysteresis

## Model Description

Two-diode model with hysteresis correction for ion-migration effects in perovskite:

```
I = I_ph - I_01*(exp(Vd/(n1*Vt))-1) - I_02*(exp(Vd/(n2*Vt))-1) - Vd/Rsh
P_actual = P_two_diode * (1 - h_factor * |dG/dt| / G_ref)
```

**Improvement over F1a:** Adds recombination diode for better accuracy and hysteresis factor for transient irradiance conditions.

## Inputs

| Parameter | Unit | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| irradiance_w_m2 | W/m2 | 0 - 1200 | - | Plane-of-array irradiance |
| temperature_cell_degC | degC | -10 to 80 | - | Cell temperature |
| irradiance_rate_w_m2_s | W/m2/s | -1000 to 1000 | 0 | Rate of irradiance change |

## Outputs

| Parameter | Unit | Description |
|-----------|------|-------------|
| i_mp | A | Current at MPP |
| v_mp | V | Voltage at MPP |
| p_mp | W | Power at MPP (after hysteresis correction) |
| efficiency | - | Cell efficiency |
| hysteresis_index | - | Fractional power loss from hysteresis |

## References

- Tress (2017). J. Phys. Chem. Lett., 8, 3106-3114.
- Miyano et al. (2016). J. Phys. Chem. Lett., 7, 2199-2202.
