# EC175 — Induction Motor/Generator — F1b Efficiency + Thermal

## Overview
Extends the F1a two-component loss model with **temperature-dependent copper resistance** and **IEC 60034-1 ambient derating**.

## Physics
- Winding resistance: `R(T) = R_ref * (1 + alpha_Cu * (T_winding - T_ref))`
- `alpha_Cu = 0.00393/K` for copper windings
- Variable (copper) losses scale with `R(T)/R_ref`
- Constant losses (iron core, friction, windage) assumed temperature-independent
- IEC 60034-1 derating: ~1%/K power reduction above 40C ambient

## Inputs
| Parameter | Unit | Default | Range |
|-----------|------|---------|-------|
| load_fraction | - | - | [0.05, 1.2] |
| winding_temperature | degC | 75 | [20, 180] |
| ambient_temperature | degC | 25 | [-20, 60] |

## Outputs
| Parameter | Unit |
|-----------|------|
| efficiency | - |
| input_power_kw | kW |
| output_power_kw | kW |
| losses_kw | kW |
| current_A | A |
| derating_factor | - |
| slip | - |

## Default Parameters
- Rated power: 7.5 kW (4-pole, IE3)
- R_ref: 0.5 ohm at 25C
- Rated efficiency: 0.917 (at 25C winding, PLR=1)

## References
- IEC 60034-30-1:2014 — Efficiency classes for motors
- IEC 60034-1:2022 — Rating and performance (thermal derating)
- Boldea, I. & Nasar, S.A. (2010). The Induction Machine Handbook. CRC Press.
