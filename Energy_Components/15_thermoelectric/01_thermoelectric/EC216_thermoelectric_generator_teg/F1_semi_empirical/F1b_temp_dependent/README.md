# EC216 — Thermoelectric Generator (TEG) — F1b Temperature-Dependent Properties Model

## Overview
Extends the F1a constant-ZT model with temperature-dependent Bi2Te3 material properties for more accurate efficiency and power predictions.

## Physics Added Over F1a
1. **Temperature-dependent Seebeck:** alpha(T) = alpha0 * (1 + a1*(T-T0) + a2*(T-T0)^2)
2. **Temperature-dependent thermal conductivity:** k(T) = k0 * (1 + b1*(T-T0))
3. **Temperature-dependent electrical conductivity:** sigma(T) = sigma0 * (1 + c1*(T-T0))
4. **Local ZT:** ZT(T) = alpha^2 * sigma * T / k — varies across the temperature gradient.
5. **Average ZT:** Integrated across T_cold to T_hot using trapezoidal rule.
6. **Contact resistance:** Included as fraction of element resistance.

## Inputs
| Parameter | Unit | Range | Default |
|-----------|------|-------|---------|
| T_hot_K | K | 323-573 | 473.15 |
| T_cold_K | K | 273-323 | 303.15 |

## Outputs
| Parameter | Unit |
|-----------|------|
| efficiency | - |
| power_density_w_cm2 | W/cm2 |
| zt_average | - |
| voltage_V | V |

## Material Properties (Bi2Te3 at T0=300K)
- alpha0 = 200 uV/K, a1 = -2e-4/K
- k0 = 1.5 W/(m*K), b1 = 3e-4/K
- sigma0 = 1e5 S/m, c1 = -5e-4/K
- n_couples = 127
- A_element = 1.4e-6 m2, L_element = 1.6e-3 m

## References
- Rowe, D.M. (ed.) (2006). Thermoelectrics Handbook. CRC Press.
- Snyder, G.J. & Toberer, E.S. (2008). Nature Materials, 7, 105-114.
