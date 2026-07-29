# EC145 Pyrolysis Reactor — F1b Feedstock Variation

## Model Summary
Semi-empirical pyrolysis model extending F1a with feedstock-specific ultimate analysis, moisture-LHV coupling, temperature-dependent product distribution, and part-load thermal efficiency.

## Key Physics
- **Moisture-LHV coupling**: `LHV_eff = LHV_dry*(1-M) - h_fg*M` (h_fg = 2442 kJ/kg)
- **Temperature-dependent yields**: Bio-oil peaks at ~500°C (Gaussian profile, σ=80°C); char decreases monotonically with T; gas is complement
- **Feedstock composition**: Cellulose/hemicellulose/lignin fractions drive bio-oil and char ratios
- **Part-load efficiency**: `eta = (a0 + a1*PLR + a2*PLR²) * moisture_factor`

## Inputs / Outputs
| Input | Unit | Range |
|-------|------|-------|
| feedstock_type | — | wood_chips, pine, corn_stover, rice_husk, switchgrass |
| temperature_degC | °C | 300–700 |
| moisture_fraction | — | 0–0.60 |
| PLR | — | 0.20–1.0 |
| feed_rate_kg_h | kg/h | 10–5000 |

| Output | Unit |
|--------|------|
| bio_oil_yield | kg/kg_dry |
| char_yield | kg/kg_dry |
| gas_yield | kg/kg_dry |
| LHV_eff_MJ_kg | MJ/kg_wet |
| moisture_lhv_factor | — |
| energy_recovery | — |
| thermal_efficiency | — |
| bio_oil/char/gas_rate_kg_h | kg/h |

## References
- Bridgwater, A.V. (2012). Biomass & Bioenergy, 38, 68-94.
- Demirbas, A. (2004). Energy Conv. Mgmt., 45(3), 653-660.
- Jenkins, B.M. et al. (1998). Fuel Processing Technology, 54, 17-46.
- Oasmaa, A. & Meier, D. (2005). J. Anal. Appl. Pyrolysis, 73(2), 323-334.
