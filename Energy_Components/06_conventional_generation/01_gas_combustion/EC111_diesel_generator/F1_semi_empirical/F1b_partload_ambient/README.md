# EC111 -- Diesel Generator -- F1b Part-Load + Ambient (Altitude & Temperature Derating)

## Overview
Extends F1a Willans-line model by adding altitude derating (3.5% per 300m above 1000m) and temperature derating (0.5% per degC above 40C). Enables site-specific performance estimation for remote/high-altitude installations.

## Model Equations
```
Willans line:
    fuel_rate [L/h] = a + b * P_elec [kW]

Altitude derating (above 1000m ASL):
    f_alt = 1 - (3.5%/300m) * max(0, altitude - 1000)

Temperature derating (above 40 degC):
    f_temp = 1 - 0.5%/degC * max(0, T_amb - 40)

Derated rated power:
    P_rated_eff = P_rated * f_alt * f_temp

Efficiency:
    eta = P_elec / (fuel_rate * rho * LHV * 1e3/3600)

SFC:
    SFC = fuel_rate * rho * 1000 / P_elec  [g/kWh]
```

## Inputs
| Name | Unit | Range | Default | Description |
|------|------|-------|---------|-------------|
| PLR | - | [0.25, 1.0] | - | Part-load ratio |
| T_ambient | degC | [-30, 55] | 25.0 | Ambient temperature |
| altitude_m | m | [0, 5000] | 0.0 | Site altitude |
| rated_power_kw | kW | [50, 5000] | 500.0 | Rated power override |

## Outputs
| Name | Unit | Description |
|------|------|-------------|
| efficiency | - | Generator efficiency |
| power_output_kw | kW | Electrical output |
| fuel_consumption_l_h | L/h | Fuel consumption |
| sfc_g_kwh | g/kWh | Specific fuel consumption |
| exhaust_temp_degC | degC | Exhaust temperature |

## Sources
1. US Army TM 5-811-6 (1996). Electric Power Plant Design.
2. Caterpillar Application and Installation Guide (2017).
3. ISO 8528-1:2018 Reciprocating IC engine driven AC generating sets.

## Limitations
- Steady-state only
- Willans line assumes linear fuel-power relationship (good for diesel at PLR > 0.25)
- Altitude derating simplified (actual depends on turbocharger design)
- No humidity correction
