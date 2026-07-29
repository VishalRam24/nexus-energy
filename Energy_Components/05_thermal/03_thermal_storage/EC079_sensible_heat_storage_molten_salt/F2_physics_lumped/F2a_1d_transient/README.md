# EC079 -- Molten Salt TES -- F2a 1D Transient Model

## Model Description
20-node 1D vertically stratified molten salt tank with temperature-dependent
Solar Salt (60% NaNO3 + 40% KNO3) properties. Operating range: 290-565 C.

## Salt Property Correlations
```
rho(T) = 2090 - 0.636*T_C  [kg/m3]
cp(T)  = 1443 + 0.172*T_C  [J/(kg.K)]
k(T)   = 0.443 + 1.9e-4*T_C [W/(m.K)]
```

## Inputs
| Parameter | Unit | Default | Description |
|-----------|------|---------|-------------|
| m_dot_charge | kg/s | 100.0 | Charge flow rate |
| T_charge_in | K | 838.15 | Hot charge temperature (565 C) |
| m_dot_discharge | kg/s | 0.0 | Discharge flow rate |
| T_discharge_in | K | 563.15 | Cold return temperature (290 C) |

## Outputs
| Parameter | Unit | Description |
|-----------|------|-------------|
| T_profiles | K | 20-node temperature profile |
| E_stored_MWh | MWh | Stored thermal energy |
| stratification_K | K | Top-bottom temperature difference |
| rho_mean, cp_mean, k_mean | various | Mean salt properties |

## References
- Zavoico (2001), SAND2001-2100
- Pacheco (2002), SAND2002-0120
