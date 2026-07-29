# EC080 -- PCM Storage -- F1b Enthalpy Method with Phase Change

## Overview
Enthalpy-method model for phase-change material (PCM) thermal storage using paraffin RT58. Builds on F1a (simple three-region) by implementing proper enthalpy tracking through the mushy zone with HTF heat exchange.

## Physics
- **Enthalpy method**: h(T) tracks total stored energy continuously through solid, mushy, and liquid regions
- **Mushy zone**: Linear interpolation of phase fraction over T_pc +/- 2 degC
- **HTF exchange**: Effectiveness-NTU model with UA_htf = 500 W/K
- **Ambient losses**: UA_loss = 5 W/K

## Parameters
| Parameter | Value | Unit |
|-----------|-------|------|
| PCM mass | 5000 | kg |
| T_pc (phase change) | 58 | degC |
| delta_T_pc | 2 | degC |
| Latent heat L | 180 | kJ/kg |
| cp_solid | 2.0 | kJ/(kg*K) |
| cp_liquid | 2.2 | kJ/(kg*K) |
| UA_htf | 500 | W/K |

## Inputs / Outputs
See predict.py get_info() for full specification.

## References
- Mehling, H. & Cabeza, L.F. (2008). Heat and Cold Storage with PCM. Springer.
- Voller, V.R. (1990). Numerical Heat Transfer B, 17(2), 155-169.
- Rubitherm Technologies GmbH -- RT58 datasheet.
