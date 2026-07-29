# EC058 — Flat Plate Solar Collector — F1a Hottel-Whillier-Bliss

## Model Card

| Field | Value |
|-------|-------|
| Component | Flat Plate Solar Collector |
| EC ID | EC058 |
| Fidelity | F1a — Hottel-Whillier-Bliss (HWB) Equation |
| Path | `03_solar/03_solar_thermal/EC058_flat_plate_solar_collector/F1_semi_empirical/F1a_hottel_whillier/` |

## Model Equations

```
Q_u = A * [F_R*(tau*alpha) * G - F_R*U_L * (T_in - T_amb)]
Q_u = max(0, Q_u)       — collector cannot cool fluid

eta = Q_u / (A * G)     — instantaneous efficiency
    = F_R*(tau*alpha) - F_R*U_L * (T_in - T_amb) / G

T_out = T_in + Q_u / (m_dot * cp)   — outlet temperature approx.
```

The HWB equation defines a linear efficiency curve in the reduced temperature parameter
`X = (T_in - T_amb) / G`, which is the standard collector characterisation used in
ISO 9806 / EN 12975 testing.

## Parameters

| Parameter | Value | Unit | Source |
|-----------|-------|------|--------|
| Area (A) | 2.0 | m² | Typical single-panel gross aperture |
| F_R*(tau*alpha) | 0.75 | — | y-intercept of efficiency curve |
| F_R*U_L | 4.5 | W/m²K | Slope of efficiency curve |
| m_dot | 0.04 | kg/s | Flow rate (0.02 kg/s/m² standard) |
| cp_fluid | 4182 | J/kgK | Water |

## Inputs / Outputs

### Inputs
| Name | Unit | Range | Description |
|------|------|-------|-------------|
| irradiance | W/m² | 0–1200 | Solar irradiance on collector plane |
| T_inlet | degC | 10–90 | Fluid inlet temperature |
| T_ambient | degC | −10–45 | Ambient air temperature |

### Outputs
| Name | Unit | Description |
|------|------|-------------|
| useful_heat_w | W | Useful heat gain |
| efficiency | — | Instantaneous collector efficiency |
| T_outlet_approx | degC | Approximate fluid outlet temperature |

## Physics Checks (all pass)
- eta <= F_R*(tau*alpha) = 0.75 (optical efficiency limit)
- eta decreases linearly with (T_in − T_amb)/G (HWB characteristic)
- Q_u = 0 at G = 0 or at stagnation (T_in too high)
- Q_u increases with G at constant temperatures
- T_out >= T_in when Q_u > 0

## Benchmark
1000 predictions in < 1 ms (NumPy vectorized)

## Limitations
- Steady-state model only — no thermal mass / transient dynamics.
- No incidence angle modifier (IAM). Use F1b for IAM correction.
- Uniform flow assumption; no stratification within collector.
- `F_R*(tau*alpha)` and `F_R*U_L` are fixed constants (no temperature dependence of U_L).
- T_outlet is approximate (uses simple energy balance; does not iterate on F_R).

## Data Sources
- Duffie, J.A. & Beckman, W.A. (2013). *Solar Engineering of Thermal Processes*, 4th ed. John Wiley & Sons, Ch. 6.

## License
BSD-3
