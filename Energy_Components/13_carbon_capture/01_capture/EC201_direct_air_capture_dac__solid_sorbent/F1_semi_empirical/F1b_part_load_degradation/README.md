# EC201 — Direct Air Capture (DAC) Solid Sorbent — F1b Part-Load + Degradation Model

## Overview
Extends the F1a energy model with sorbent degradation, humidity-dependent capacity, and temperature swing energy.

## Physics Added Over F1a
1. **Sorbent degradation:** q(n) = q0 * (1 - k_deg * n_cycles), k_deg = 5e-5/cycle.
2. **Humidity effect:** Capacity factor = f(RH) via quadratic fit. Low humidity reduces amine sorbent performance.
3. **Temperature swing energy:** E_th includes sensible heat m*cp*dT plus desorption enthalpy.
4. **Part-load:** Reduced air flow with fan energy penalty.

## Inputs
| Parameter | Unit | Range | Default |
|-----------|------|-------|---------|
| air_flow_m3_s | m3/s | 1-100 | 10 |
| T_ambient_degC | degC | -10 to 45 | 20 |
| relative_humidity | - | 0.2-0.8 | 0.5 |
| PLR | - | 0.3-1.0 | 1.0 |
| n_cycles | cycles | 0-100000 | 0 |

## Outputs
| Parameter | Unit |
|-----------|------|
| co2_captured_kg_h | kg/h |
| thermal_energy_kwh_ton | kWh_th/tCO2 |
| electrical_energy_kwh_ton | kWh_e/tCO2 |
| sorbent_capacity_pct | % |

## Key Parameters
- q0 = 1.5 mmol_CO2/g_sorbent
- sorbent_mass = 10,000 kg
- T_desorption = 100 degC
- cycle_time = 3600 s
- k_deg = 5e-5 /cycle

## References
- Fasihi et al. (2019). J. Cleaner Production, 224, 957-980.
- Sinha, A. et al. (2017). Ind. Eng. Chem. Res., 56(3), 750-764.
