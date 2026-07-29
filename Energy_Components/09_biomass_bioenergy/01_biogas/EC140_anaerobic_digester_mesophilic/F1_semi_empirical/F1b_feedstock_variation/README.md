# EC140 -- Anaerobic Digester -- F1b Feedstock-Specific BMP with Co-Digestion

## Overview
Feedstock-specific biochemical methane potential (BMP) model with co-digestion synergy and C/N ratio optimization. Builds on F1a (single yield model) by supporting 5 feedstocks with unique BMPs, hydrolysis rates, and methane content.

## Physics
- **BMP database**: Feedstock-specific methane potential (L CH4/kgVS)
- **First-order kinetics**: Yield = BMP * (1 - exp(-k*HRT)) with feedstock-specific k
- **C/N ratio**: Computed from feedstock carbon/nitrogen content; penalty outside [20,30]
- **Co-digestion synergy**: Up to 10% yield bonus when blending optimizes C/N ratio
- **Temperature correction**: Arrhenius model normalized to 37 degC reference

## Feedstock Database
| Feedstock | BMP (L CH4/kgVS) | C/N | k_hydrolysis (1/day) |
|-----------|-------------------|-----|---------------------|
| cattle_manure | 250 | 14.0 | 0.10 |
| food_waste | 400 | 15.0 | 0.25 |
| grass_silage | 350 | 23.0 | 0.12 |
| sewage_sludge | 200 | 6.7 | 0.08 |
| corn_silage | 340 | 34.6 | 0.14 |

## References
- Angelidaki, I. et al. (2009). Water Science & Technology, 59(5), 927-934.
- Mata-Alvarez, J. et al. (2014). Renewable and Sustainable Energy Reviews, 36, 412-427.
