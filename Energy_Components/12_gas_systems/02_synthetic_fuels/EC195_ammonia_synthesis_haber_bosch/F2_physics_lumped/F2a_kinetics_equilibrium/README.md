# EC195 -- Ammonia Synthesis (Haber-Bosch) -- F2a Temkin-Pyzhev + CSTR

## Model Description

Physics-lumped CSTR model for ammonia synthesis via the Haber-Bosch process. Uses Temkin-Pyzhev
kinetics with temperature-dependent equilibrium constant and coupled energy balance. Includes
a recycle loop simulation for overall plant conversion estimation.

## Physics

**Reaction:**
```
N2 + 3H2 <-> 2NH3    (delta_H = -92 kJ/mol)
```

**Temkin-Pyzhev rate:**
```
r = k_f * K_eq^0.5 * (P_N2 * P_H2^1.5 / P_NH3) - k_f / K_eq^0.5 * (P_NH3 / P_H2^1.5)
k_f = 8.85e14 * exp(-170000/(R*T))
```

**Equilibrium constant (Gillespie-Beattie):**
```
ln(K_eq) = -2.691122*ln(T) - 5.519265e-5*T + 1.848863e-7*T^2 + 2001.6/T + 2.6899
```

## Inputs

| Parameter | Unit | Default | Range |
|-----------|------|---------|-------|
| T0_K | K | 673.15 | 573-873 |
| P_atm | atm | 200 | 100-350 |
| GHSV | 1/h | 10000 | 5000-30000 |
| T_cool_K | K | 673.15 | 573-773 |
| with_recycle | bool | False | - |

## Outputs

| Parameter | Unit |
|-----------|------|
| T | K (temperature time series) |
| X_N2 | - (N2 conversion) |
| y_NH3 | - (NH3 mole fraction) |
| C_N2, C_H2, C_NH3 | mol/m3 |
| overall_conversion | - (with recycle) |
| energy_per_ton_NH3_GJ | GJ/ton (with recycle) |

## Features

- Single-pass CSTR dynamics with BDF stiff solver
- Recycle loop simulation (iterative passes)
- Equilibrium conversion vs T and P analysis
- Energy per ton NH3 estimation

## References

- Temkin & Pyzhev (1940) Acta Physicochim. URSS
- Gillespie & Beattie (1930) Phys. Rev.
- Appl (1999) Ammonia: Principles and Industrial Practice, Wiley-VCH

## Limitations

- CSTR assumption (real converters are multi-bed adiabatic PFR)
- Simplified Temkin-Pyzhev (not the full fugacity-based form)
- Ideal gas law (real Haber-Bosch operates at high P, non-ideal)
- Recycle loop is iterative steady-state, not fully dynamic
