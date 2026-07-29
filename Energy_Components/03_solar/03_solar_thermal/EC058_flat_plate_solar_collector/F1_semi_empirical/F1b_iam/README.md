# EC058 — Flat Plate Solar Collector — F1b IAM

## Model Description

Hottel-Whillier-Bliss model with Incidence Angle Modifier:

```
IAM(theta) = 1 - b0 * (1/cos(theta) - 1)
Q_u = A * F_R * [IAM * tau_alpha * G - U_L * (T_in - T_amb)]
```

**Improvement over F1a:** Adds angular dependence of optical performance via the ASHRAE IAM model. More accurate for real installations where incidence angle varies throughout the day.

## Inputs

| Parameter | Unit | Range | Description |
|-----------|------|-------|-------------|
| irradiance_w_m2 | W/m2 | 0 - 1200 | Irradiance on collector plane |
| incidence_angle_deg | deg | 0 - 85 | Angle of incidence |
| T_inlet_degC | degC | 10 - 90 | Fluid inlet temperature |
| T_ambient_degC | degC | -10 to 45 | Ambient temperature |

## Outputs

| Parameter | Unit | Description |
|-----------|------|-------------|
| thermal_output_w | W | Useful heat gain |
| efficiency | - | Collector efficiency |
| iam_factor | - | Incidence angle modifier |
| T_outlet_degC | degC | Fluid outlet temperature |

## Parameters

| Parameter | Value | Unit | Description |
|-----------|-------|------|-------------|
| A | 2.0 | m2 | Collector area |
| F_R | 0.80 | - | Heat removal factor |
| tau_alpha | 0.82 | - | Normal-incidence transmittance-absorptance |
| U_L | 3.5 | W/m2K | Overall loss coefficient |
| b0 | 0.15 | - | IAM coefficient |

## References

- Duffie & Beckman (2013), Solar Engineering of Thermal Processes, Ch.6.
- ASHRAE Standard 93 (2010).
