# EC054 — Parabolic Trough CSP — F1a Optical Efficiency + Heat Loss

## Model Card

| Field | Value |
|-------|-------|
| Component | Parabolic Trough Concentrating Solar Power |
| EC ID | EC054 |
| Fidelity | F1a — Optical Efficiency + Receiver Heat Loss |
| Path | `03_solar/02_concentrated_solar_power/EC054_parabolic_trough_csp/F1_semi_empirical/F1a_optical_eta/` |

## Model Equations

```
eta_optical = reflectivity * intercept_factor * transmissivity * absorptivity
            = 0.93 * 0.92 * 0.95 * 0.92 ≈ 0.75

IAM(theta)  = 1 - 0.0003 * theta^2          (incidence angle modifier)

Q_absorbed  = DNI * A_aperture * eta_optical * IAM(theta)   [W -> kW]

q_loss/m    = a0 + a1*(T_abs - T_amb) + a2*(T_abs - T_amb)^2    [W/m, Schott PTR70]
Q_loss      = q_loss/m * L_collector                              [kW]

Q_useful    = max(0, Q_absorbed - Q_loss)

eta_overall = Q_useful / (DNI * A_aperture)
```

## Parameters

| Parameter | Value | Unit | Source |
|-----------|-------|------|--------|
| Aperture width | 5.76 | m | LS-3 collector |
| L_collector | 150 | m | Single loop length |
| A_aperture | 864 | m² | = 5.76 × 150 |
| eta_optical | 0.75 | — | Peak optical efficiency |
| IAM_coeff | 0.0003 | 1/deg² | Cosine-squared IAM fit |
| a1 (heat loss) | 0.06 | W/(m·K) | Schott PTR70 receiver, linear term |
| a2 (heat loss) | 0.0001 | W/(m·K²) | Schott PTR70, quadratic term |

## Inputs / Outputs

### Inputs
| Name | Unit | Range | Description |
|------|------|-------|-------------|
| dni | W/m² | 0–1000 | Direct Normal Irradiance |
| T_absorber | degC | 100–400 | HTF / absorber tube mean temperature |
| T_ambient | degC | 0–50 | Ambient temperature |
| incidence_angle | deg | 0–80 | Angle between sun vector and collector normal |

### Outputs
| Name | Unit | Description |
|------|------|-------------|
| useful_heat_kw | kW | Net useful thermal output |
| optical_efficiency | — | Optical efficiency including IAM |
| thermal_loss_kw | kW | Receiver heat loss (conduction + radiation) |
| overall_efficiency | — | Q_useful / (DNI × A_aperture) |

## Physics Checks (all pass)
- Q_useful < DNI × A × eta_opt (heat losses reduce output)
- eta_overall <= eta_optical at all conditions
- Q_loss increases with T_absorber (quadratic)
- IAM = 1 at theta = 0 (normal incidence)
- IAM decreases with increasing theta
- Q_useful = 0 at DNI = 0
- Q_useful >= 0 (clamped)
- Q_useful increases monotonically with DNI

## Rated Performance
At DNI=850 W/m², T_abs=300°C, T_amb=25°C, theta=0°:
- Q_absorbed ≈ 551 kW, Q_loss ≈ 41 kW, Q_useful ≈ 510 kW
- eta_overall ≈ 0.694

## Benchmark
1000 predictions in < 1 ms (NumPy vectorized)

## Limitations
- 1D heat loss model (Schott PTR70 empirical correlation). For annulus vacuum degradation, use F1b.
- IAM polynomial is a first-order approximation; full IAM uses shade + end-loss corrections.
- HTF temperature is treated as uniform along the loop (no 1D spatial distribution). Use F2 for distributed model.
- No thermal inertia. No HTF pressure drop. No pump work.

## Data Sources
- Forristall, R. (2003). *Heat Transfer Analysis and Modeling of a Parabolic Trough Solar Receiver Implemented in Engineering Equation Solver*. NREL/TP-550-34169. National Renewable Energy Laboratory.

## License
BSD-3
