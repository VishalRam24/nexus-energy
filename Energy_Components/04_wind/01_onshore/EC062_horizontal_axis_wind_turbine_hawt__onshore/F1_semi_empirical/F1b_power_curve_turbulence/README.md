# EC062 — HAWT Onshore Wind Turbine — F1b Turbulence Correction

## Model Description

Power curve with second-order turbulence correction:

```
P_corr(V) = P(V) + 0.5 * d2P/dV2 * sigma_v^2
sigma_v = TI * V
```

Uses cubic spline interpolation of the manufacturer power curve to obtain smooth second derivatives. The correction accounts for the non-linear relationship between wind speed and power.

**Improvement over F1a:** Adds turbulence intensity as an input, correcting for the effect of velocity fluctuations on mean power output. Critical for onshore sites with TI = 0.10-0.25.

## Inputs

| Parameter | Unit | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| wind_speed_m_s | m/s | 0 - 30 | - | Hub-height wind speed |
| turbulence_intensity | - | 0 - 0.30 | 0 | TI = sigma_v / V |
| air_density | kg/m3 | 0.9 - 1.4 | 1.225 | Air density |

## Outputs

| Parameter | Unit | Description |
|-----------|------|-------------|
| power_kw | kW | Turbulence-corrected power |
| power_coefficient | - | Cp |
| capacity_factor_correction | - | CF change from turbulence |

## References

- Albers et al. (2007). Wind Energy, 10(4), 395-406.
- IEC 61400-12-1:2017.
