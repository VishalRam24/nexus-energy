# EC209 — Reverse Osmosis (RO) — F1a Specific Energy Consumption Model

## Overview
Semi-empirical SEC model for RO desalination. Computes feed pressure from osmotic pressure
(van't Hoff approximation) and accounts for energy recovery device (ERD) on the brine stream.

## Model Equations
```
osmotic_pressure = 0.7 * S_feed                           [bar, S in g/L]
P_feed           = osmotic_pressure / recovery + dP_mem   [bar]
P_brine          = P_feed - dP_mem                        [bar]
SEC_net          = (P_feed/eta_pump - P_brine*(1-r)*eta_ERD) / (r * 36)  [kWh/m3]
permeate_flow    = feed_flow * recovery                   [m3/hr]
permeate_salinity = S_feed * (1 - rejection)              [g/L]
```

## Inputs
| Name | Unit | Range | Description |
|------|------|-------|-------------|
| feed_salinity | g/L | [1, 45] | Feed water total dissolved solids |
| recovery | - | [0.2, 0.6] | Water recovery fraction |
| feed_flow_m3h | m3/hr | [10, 1000] | Feed volumetric flow rate |

## Outputs
| Name | Unit | Description |
|------|------|-------------|
| sec_kwhm3 | kWh/m3 | Specific energy consumption (per m3 permeate) |
| permeate_flow_m3h | m3/hr | Product water flow rate |
| feed_pressure_bar | bar | Required high-pressure pump delivery pressure |
| permeate_salinity_gl | g/L | Product water TDS |

## Parameters
| Parameter | Value | Unit | Source |
|-----------|-------|------|--------|
| eta_pump | 0.85 | - | Typical SWRO pump |
| eta_ERD | 0.95 | - | Pressure exchanger type |
| dP_membrane | 2.0 | bar | Typical hollow-fiber/spiral-wound |
| pi_coeff | 0.7 | bar/(g/L) | Simplified van't Hoff for NaCl |
| salt_rejection | 0.995 | - | Typical SWRO membrane |

## Sources
1. Elimelech, M. & Phillip, W. A. (2011). The future of seawater desalination: Energy,
   technology, and the environment. *Science*, 333, 712-717.

## Limitations
- Osmotic pressure linearized (no temperature or ion-activity corrections)
- No fouling or concentration polarization effects
- Constant rejection assumed (independent of flux)
- No boron or other trace contaminant removal model
