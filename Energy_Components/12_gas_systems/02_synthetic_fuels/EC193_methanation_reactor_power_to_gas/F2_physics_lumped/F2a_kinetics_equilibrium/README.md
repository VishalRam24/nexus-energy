# EC193 -- Methanation Reactor (Power-to-Gas) -- F2a Kinetics + Equilibrium CSTR

## Model Description

Physics-lumped CSTR model for catalytic CO2 methanation (Sabatier reaction) with coupled
kinetics and energy balance. Uses `scipy.integrate.solve_ivp` (BDF method) for stiff
chemical kinetics ODEs.

## Physics

**Sabatier reaction:**
```
CO2 + 4H2 -> CH4 + 2H2O    (delta_H = -165 kJ/mol)
```

**Rate law (power law):**
```
r = k * P_CO2^0.5 * P_H2^0.5   [mol/(kg_cat.s)]
k = A * exp(-Ea/(R*T)),  A = 1.5e7, Ea = 85 kJ/mol
```

**CSTR mole balance:**
```
dC_i/dt = (C_i_in - C_i)/tau + nu_i * r * rho_cat_eff
```

**Energy balance:**
```
(m*cp) * dT/dt = (-dH)*r*m_cat - UA*(T-T_cool) + F_in*cp*(T_in-T)
```

## Inputs

| Parameter | Unit | Default | Range |
|-----------|------|---------|-------|
| T0_K | K | 523.15 | 473-873 |
| P_bar | bar | 10.0 | 1-30 |
| GHSV | 1/h | 3000 | 1000-10000 |
| T_cool_K | K | 523.15 | 473-673 |
| duration_s | s | 600 | - |

## Outputs

| Parameter | Unit |
|-----------|------|
| T | K (temperature time series) |
| X_CO2 | - (CO2 conversion) |
| y_CH4_dry | - (dry CH4 mole fraction) |
| C_CO2, C_H2, C_CH4, C_H2O | mol/m3 |
| thermal_runaway | bool |
| X_eq_final | - (equilibrium limit) |

## Features

- Equilibrium conversion limit check
- Thermal runaway detection (T > T_in + 300 K)
- Steady-state conversion vs temperature analysis
- Pressure effect on conversion

## References

- Koschany et al. (2016) Applied Catalysis B, 181, 504-516
- Roensch et al. (2016) Fuel, 166, 276-296

## Limitations

- CSTR (perfectly mixed) assumption -- real reactors are PFR-like
- Simplified power-law kinetics (not LHHW)
- No catalyst deactivation
- Ideal gas assumed
