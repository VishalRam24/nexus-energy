# EC054 — Parabolic Trough CSP — F1b Receiver Losses

## Model Description

Detailed receiver heat loss model with physics-based convective and radiative losses:

```
Q_loss/L = pi * D_abs * [h_conv*(T_abs - T_amb) + eps*sigma*(T_abs^4 - T_sky^4)]
IAM(theta) = cos(theta) - 5.25097e-4*theta - 2.859621e-5*theta^2
f_end = 1 - f_L*tan(theta) / L_collector
```

**Improvement over F1a:** Replaces polynomial heat loss fit with first-principles convective + radiative model. Adds end loss factor and Sandia polynomial IAM.

## Inputs

| Parameter | Unit | Range | Description |
|-----------|------|-------|-------------|
| DNI_w_m2 | W/m2 | 0 - 1000 | Direct Normal Irradiance |
| T_htf_in_degC | degC | 100 - 400 | HTF inlet temperature |
| T_htf_out_degC | degC | 100 - 400 | HTF outlet temperature |
| T_ambient_degC | degC | 0 - 50 | Ambient temperature |
| incidence_angle_deg | deg | 0 - 80 | Sun incidence angle |

## Outputs

| Parameter | Unit | Description |
|-----------|------|-------------|
| thermal_output_kw_per_m | kW/m | Net thermal output per receiver length |
| optical_efficiency | - | Total optical efficiency |
| thermal_efficiency | - | Net thermal / incident solar |
| receiver_loss_kw_per_m | kW/m | Thermal loss per receiver length |

## References

- Forristall (2003), NREL/TP-550-34169.
- Dudley et al. (1994), SAND94-1884.
- Kalogirou (2012), Solar Energy 86(1), 1-17.
