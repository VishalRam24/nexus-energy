# EC111 — Diesel Generator — F1a Willans Line Model

## Overview
Semi-empirical Willans line model for diesel generator fuel consumption and efficiency. The fuel rate is linear with power output, parameterized by a no-load intercept and an incremental slope.

## Model Equations
```
fuel_rate   = a + b * P_out                    [L/h]
eta         = P_out / (fuel_rate * rho * LHV / 3.6)
SFC         = fuel_rate * rho * 1000 / P_out  [g/kWh]
CO2_rate    = fuel_rate * co2_factor           [kg_CO2/h]
```

## Inputs
| Name | Unit | Range | Description |
|------|------|-------|-------------|
| power_output_kw | kW | [0, 500] | Electrical power output |
| ambient_temp_c | degC | [-20, 55] | Ambient temperature (for derating, default 25) |

## Outputs
| Name | Unit | Description |
|------|------|-------------|
| fuel_rate_lph | L/h | Diesel fuel consumption rate |
| sfc_gkwh | g/kWh | Specific fuel consumption |
| efficiency | - | Generator efficiency (P_out/fuel_energy_in) |
| co2_emissions_kgh | kg_CO2/h | CO2 emission rate |

## Parameters (500 kW unit)
| Parameter | Value | Unit | Source |
|-----------|-------|------|--------|
| P_rated | 500 | kW | Nameplate |
| a (no-load) | 10 | L/h | TM 5-811-6 |
| b (incremental) | 0.25 | L/kWh | TM 5-811-6 |
| SFC at rated | ~210 | g/kWh | Nameplate |
| rho_diesel | 0.832 | kg/L | ASTM D975 |
| LHV_diesel | 42.5 | MJ/kg | IPCC 2006 |
| PLR_min | 0.25 | - | Standard practice |
| CO2 factor | 2.68 | kg_CO2/L | IPCC 2006 |

## Sources
1. US Army TM 5-811-6 (1996). Electric Power Plant Design.
2. Tuffaha, M. & Gravdahl, J.T. (2014). Modeling and control of diesel generators in a microgrid. IEEE MELECON.
3. IPCC (2006). Emission factors for stationary combustion — diesel.

## Physics Checks
- eta < 0.45 across all loads (thermodynamic limit)
- SFC decreases (improves) as load increases toward rated
- fuel_rate > 0 even at no-load (engine idling)
- CO2 proportional to fuel consumption

## Limitations
- No transient dynamics (steady-state Willans line only)
- No degradation modeling
- No partial-load COP correction beyond Willans slope
- Derating is linear above 25 degC (simplified)
