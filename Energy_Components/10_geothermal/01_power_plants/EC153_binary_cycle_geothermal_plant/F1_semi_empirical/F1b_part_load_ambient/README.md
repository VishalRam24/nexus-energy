# EC153 -- Binary Cycle Geothermal Plant -- F1b Part-Load & Ambient Derating

## Overview
Part-load ORC efficiency model with air-cooled condenser ambient derating and resource temperature decline. Builds on F1a (exergy model) by adding realistic operational derating factors.

## Physics
- **Part-load curve**: Empirical eta_ratio vs PLR lookup (efficiency drops at low loads)
- **Ambient derating**: Air-cooled condenser loses performance at high ambient temperatures; f_cond = 1 - 0.005*(T_amb - T_cond_design)
- **Resource decline**: Brine temperature decreases at 1.5%/year; f_resource = (1-0.015)^years
- **Combined derating**: P = Q_in * eta_design * f_PLR * f_cond * carnot_ratio * PLR

## Parameters
| Parameter | Value | Unit |
|-----------|-------|------|
| P_rated | 5000 | kW |
| T_brine_design | 150 | degC |
| T_cond_design | 30 | degC |
| eta_design | 0.12 | - |
| decline_rate | 1.5 | %/yr |
| k_ambient | 0.005 | 1/degC |

## References
- DiPippo, R. (2015). Geothermal Power Plants, 4th ed. Butterworth-Heinemann.
- Lukawski, M.Z. et al. (2014). Proceedings, 39th Workshop on Geothermal Reservoir Engineering.
