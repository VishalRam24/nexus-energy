# EC044 — Monocrystalline Silicon PV — F1b Two-Diode Model

## Model Description

Two-diode model separating diffusion and recombination currents for improved low-irradiance accuracy:

```
I = I_ph - I_01*(exp((V+I*Rs)/(n1*Vt))-1) - I_02*(exp((V+I*Rs)/(n2*Vt))-1) - (V+I*Rs)/Rsh
```

where `Vt = N_s * k * T / q` is the module thermal voltage.

**Improvement over F1a:** The second diode (n2=2) captures space-charge recombination effects that dominate at low irradiance, providing better fill factor and efficiency predictions below 200 W/m2.

## Inputs

| Parameter | Unit | Range | Description |
|-----------|------|-------|-------------|
| irradiance_w_m2 | W/m2 | 0 - 1200 | Plane-of-array irradiance |
| temperature_cell_degC | degC | -10 to 80 | Cell temperature |

## Outputs

| Parameter | Unit | Description |
|-----------|------|-------------|
| i_mp | A | Current at maximum power point |
| v_mp | V | Voltage at maximum power point |
| p_mp | W | Maximum power |
| i_sc | A | Short-circuit current |
| v_oc | V | Open-circuit voltage |
| fill_factor | - | Fill factor = Pmp / (Voc * Isc) |
| efficiency | - | Module efficiency = Pmp / (G * A) |

## Parameters

| Parameter | Value | Unit | Description |
|-----------|-------|------|-------------|
| I_ph_ref | 9.5 | A | Photocurrent at STC |
| I_01 | 1e-12 | A | Diffusion saturation current |
| I_02 | 1e-7 | A | Recombination saturation current |
| n1 | 1.0 | - | Ideality factor, diode 1 |
| n2 | 2.0 | - | Ideality factor, diode 2 |
| Rs | 0.3 | Ohm | Series resistance |
| Rsh | 300 | Ohm | Shunt resistance |
| N_s | 60 | cells | Cells in series |

## References

- Ishaque et al. (2011). Solar Energy, 85(9), 2349-2359.
- De Soto et al. (2006). Solar Energy, 80(1), 78-88.

## Limitations

- Numerical solver (Brent's method) is slower than F1a analytical Lambert-W solution
- Parameter extraction requires fitting to measured I-V curves for best accuracy
- Does not include thermal or degradation effects (see F1c, F1d)
