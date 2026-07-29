# EC221 — MHD Generator — F1b Hall Parameter Model

**Fidelity:** F1b — Semi-Empirical  
**Sub-branch:** Hall parameter correction + temperature-dependent conductivity + stagnation-enthalpy Q_in

## Model Description

Extends F1a (ideal Faraday MHD) with three physics upgrades:

| Upgrade | F1a | F1b |
|---------|-----|-----|
| Heat input Q_in | kinetic `0.5*rho*u^3*A` (wrong) | stagnation enthalpy `rho*u*(cp*T + 0.5*u^2)*A` (correct) |
| Conductivity | constant `sigma` | temperature-dependent `sigma(T) ~ T^1.5` (Spitzer) |
| Cross-field losses | none | Hall parameter `sigma_eff = sigma/(1+beta^2)` |

**Critical physics note — Phase 7 catch:** F1a used Q_in = 0.5*rho*u^3*A (pure kinetic energy), which violates the first law for thermally-dominated plasma. At T=2500K and u=800 m/s: cp*T ~ 3.0 MJ/kg vs 0.5*u^2 ~ 0.32 MJ/kg. The stagnation enthalpy is ~10x larger than kinetic energy alone, so F1a's eta_plant was systematically over-estimated by ~10x.

## Inputs

| Name | Unit | Default | Range |
|------|------|---------|-------|
| sigma | S/m | 10.0 | 1–100 |
| u | m/s | 800 | 100–2000 |
| B | T | 5.0 | 0.5–10 |
| K | - | 0.5 | 0–1 |
| beta | - | 3.0 | 0–10 |
| T_plasma_K | K | 2500 | 1500–4000 |

## Outputs

| Name | Unit | Description |
|------|------|-------------|
| EMF_V | V | Induced EMF per channel |
| J_Am2 | A/m² | Faraday current density |
| J_hall_Am2 | A/m² | Hall current density (perpendicular) |
| sigma_eff_Sm | S/m | Hall-corrected effective conductivity |
| power_elec_W | W | Total electrical output |
| heat_input_stag_W | W | Stagnation enthalpy flux (Q_in) |
| eta_mhd | - | Ideal Faraday fraction K(1-K) |
| eta_hall | - | Hall reduction factor 1/(1+beta²) |
| eta_electric | - | First-law efficiency P/Q_in |

## Physics Notes

- **Optimal load factor K_opt = 0.5** analytically, regardless of Hall parameter (sigma_eff factors out of the K-optimization)
- **Hall effect** reduces power by 1/(1+beta^2): at beta=3, power is 10× lower than ideal
- **Conductivity scaling** sigma ~ T^1.5 (Spitzer regime, seeded combustion plasma)
- Channel geometry: 5m × 0.5m × 0.5m; superconducting B=5T

## References

- Rosa, R.J. (1987). *Magnetohydrodynamic Energy Conversion*. McGraw-Hill.
- Messerle, H.K. (1995). *MHD Electrical Power Generation*. Wiley.
- Veefkind, A. (1977). Hall parameter effects. *Appl. Phys. Lett.*
