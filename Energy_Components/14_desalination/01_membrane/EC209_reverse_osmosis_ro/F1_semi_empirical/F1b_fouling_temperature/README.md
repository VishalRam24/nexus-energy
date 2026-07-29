# EC209 — Reverse Osmosis (RO) — F1b Fouling + Temperature Model

## Overview
Extends the F1a SEC model with membrane fouling degradation and temperature-dependent water flux.

## Physics Added Over F1a
1. **Membrane fouling:** A(t) = A0 * exp(-k_foul * t/8760), k_foul = 0.1/year. Exponential permeability decline.
2. **Temperature-dependent flux:** J(T) = J_ref * exp(2500*(1/T_ref - 1/T)). Arrhenius-type viscosity correction.
3. **Osmotic pressure:** pi = 0.7 * S/1000 bar (S in ppm). With concentration polarization at membrane surface.
4. **Salt rejection degradation:** Slight decrease with membrane aging.

## Inputs
| Parameter | Unit | Range | Default |
|-----------|------|-------|---------|
| feed_salinity_ppm | ppm | 1000-45000 | 35000 |
| feed_pressure_bar | bar | 20-80 | 60 |
| feed_temperature_degC | degC | 10-40 | 25 |
| recovery_ratio | - | 0.3-0.6 | 0.45 |
| operating_hours | hours | 0-87600 | 0 |

## Outputs
| Parameter | Unit |
|-----------|------|
| permeate_flow_m3_h | m3/h |
| sec_kwh_m3 | kWh/m3 |
| rejection_pct | % |
| flux_decline_factor | - |

## Key Parameters
- A0 = 4.0 L/(m2*h*bar) (clean membrane permeability)
- membrane_area = 37 m2
- B = 0.05 L/(m2*h)
- pump_efficiency = 0.85
- k_foul = 0.10/year

## References
- Elimelech, M. & Phillip, W. A. (2011). Science, 333, 712-717.
- Kang, G. & Cao, Y. (2012). Water Research, 46(3), 584-600.
