# EC143 -- Biomass Gasifier -- F1b Feedstock-Specific Model

## Overview
Feedstock-specific gasification model with ultimate analysis (C,H,O,N,S) driving syngas composition predictions. Builds on F1a (single equilibrium) by supporting 5 feedstocks with moisture correction, tar estimation, and feedstock-dependent efficiency.

## Physics
- **Ultimate analysis**: Each feedstock has C, H, O, N, S, ash mass fractions and HHV
- **ER effect**: Higher equivalence ratio -> more CO2, less CO (more oxidation)
- **Moisture correction**: Higher moisture -> lower gasification efficiency, H2 shift
- **Tar model**: Exponential decay with ER (more oxidation breaks tars)
- **Cold gas efficiency**: CGE = LHV_syngas * V_syngas / HHV_biomass

## Feedstock Database
| Feedstock | C | H | O | Ash | HHV (MJ/kg) |
|-----------|---|---|---|-----|-------------|
| wood | 0.50 | 0.06 | 0.42 | 0.01 | 20.0 |
| rice_husk | 0.38 | 0.05 | 0.36 | 0.19 | 15.0 |
| pine | 0.52 | 0.06 | 0.40 | 0.005 | 21.0 |
| corn_stover | 0.44 | 0.06 | 0.42 | 0.055 | 17.5 |
| sewage_sludge | 0.30 | 0.05 | 0.20 | 0.35 | 12.0 |

## References
- Zainal, Z.A. et al. (2001). Energy Conversion and Management, 42(12), 1499-1515.
- Basu, P. (2010). Biomass Gasification and Pyrolysis. Academic Press.
- Li, X.T. et al. (2004). Biomass & Bioenergy, 26(2), 171-193.
