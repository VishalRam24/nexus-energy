# EC065 — Offshore Fixed-Bottom Wind Turbine — F1b Turbulence + Humid Density

## Model Description

Power curve with turbulence correction and humid air density correction:

```
P_corr(V) = P(V) + 0.5 * d2P/dV2 * sigma_v^2    (turbulence)
rho = P_atm / (Rd * T) * (1 - 0.378 * e_w / P_atm)  (humid air)
e_w = RH * 611.21 * exp(17.502*T / (240.97 + T))     (Buck equation)
```

**Improvement over F1a:** Adds turbulence correction and physics-based humid air density. Offshore environments have lower TI (0.06-0.10) but significant humidity effects.

## Inputs

| Parameter | Unit | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| wind_speed_m_s | m/s | 0 - 30 | - | Hub-height wind speed |
| turbulence_intensity | - | 0 - 0.30 | 0.06 | TI = sigma_v / V |
| air_temperature_degC | degC | -10 to 40 | 15 | Air temperature |
| relative_humidity | - | 0 - 1 | 0.5 | Relative humidity |

## Outputs

| Parameter | Unit | Description |
|-----------|------|-------------|
| power_kw | kW | Corrected power output |
| power_coefficient | - | Cp |
| air_density_corrected | kg/m3 | Humid air density |

## References

- Albers et al. (2007). Wind Energy, 10(4), 395-406.
- Buck (1981). J. Appl. Meteorol. Climatol., 20(12), 1527-1532.
- IEC 61400-12-1:2017.
