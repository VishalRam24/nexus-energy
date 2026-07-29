# EC201 — Direct Air Capture (DAC) Solid Sorbent — F1a Energy Model

## Overview
Semi-empirical energy consumption model for solid-sorbent DAC systems. Captures the dependence
of thermal energy on sorbent capture efficiency and ambient humidity, plus constant electrical
energy for fans and vacuum.

## Model Equations
```
humidity_factor = 1 + 0.3 * (RH - 0.5)
E_thermal       = E_th_base / (capture_efficiency * humidity_factor)   [kWh_th/tCO2]
E_electric      = E_el_base                                            [kWh_e/tCO2]
CO2_conc_mass   = 415e-6 * (44/29) * rho_air                          [kg_CO2/m3_air]
capture_rate    = air_flow * CO2_conc_mass * capture_efficiency        [kg/s]
co2_tpa         = capture_rate * 3600 * 8760 / 1000                   [tCO2/yr]
```

## Inputs
| Name | Unit | Range | Description |
|------|------|-------|-------------|
| air_flow_m3h | m3/hr | [1e5, 1e7] | Air throughput |
| relative_humidity | - | [0.1, 0.9] | Ambient relative humidity |
| ambient_temp | degC | [-10, 45] | Ambient temperature (reserved) |

## Outputs
| Name | Unit | Description |
|------|------|-------------|
| co2_captured_tpa | tCO2/yr | Annual CO2 captured |
| thermal_energy_mwh_pa | MWh_th/yr | Annual thermal energy input |
| electrical_energy_mwh_pa | MWh_e/yr | Annual electrical energy input |
| specific_thermal_kwht | kWh_th/tCO2 | Specific thermal energy |
| specific_electric_kwhe | kWh_e/tCO2 | Specific electrical energy |

## Parameters
| Parameter | Value | Unit | Source |
|-----------|-------|------|--------|
| E_th_base | 1500 | kWh_th/tCO2 | Fasihi et al. (2019) |
| E_el_base | 250 | kWh_e/tCO2 | Fasihi et al. (2019) |
| T_regen | 100 | degC | Literature |
| capture_efficiency | 0.75 | - | Fasihi et al. (2019) |
| CO2 concentration | 415 | ppm_v | Global avg (2019) |

## Sources
1. Fasihi, M., Efimova, O., & Breyer, C. (2019). Techno-economic assessment of CO2 direct air
   capture plants. *Journal of Cleaner Production*, 224, 957-980.

## Limitations
- No temperature dependence of sorbent capacity (T_amb reserved for future sub-fidelity)
- No part-load degradation model
- Assumes 8760 operating hours/year (continuous operation)
- CO2 concentration fixed at 415 ppm (2019 global average)
- Humidity factor is a simplified linear approximation
