# EC104 -- Gas Engine CHP -- F2a Otto Cycle Model

## Model Description
Air-standard Otto cycle thermodynamic model with heat recovery.

**Cycle:** 1->2 isentropic compression, 2->3 constant volume heat addition, 3->4 isentropic expansion, 4->1 constant volume heat rejection.

**Efficiency:** `eta_Otto = 1 - 1/r^(gamma-1)`

**Heat recovery:** exhaust gas + jacket water cooling.

## Inputs
| Parameter | Unit | Default | Range |
|-----------|------|---------|-------|
| fuel_input_kw | kW | - | [50, 5000] |
| compression_ratio | - | 12 | [8, 16] |
| T_ambient_K | K | 298.15 | [253, 323] |

## Outputs
| Parameter | Unit |
|-----------|------|
| power_electrical_kw | kW |
| heat_exhaust_kw | kW |
| heat_jacket_kw | kW |
| eta_electrical | dimensionless |
| eta_thermal | dimensionless |
| T_exhaust_K | K |

## References
- Cengel & Boles (2019), Thermodynamics, 9th ed.
- US EPA CHP Technology Fact Sheets (2017).
