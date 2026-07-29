# EC122 — Pumped Hydro Storage (PHS) — F1a Round-Trip Model

## Overview
Semi-empirical round-trip efficiency model for pumped hydro storage. Covers turbine-generator
(discharge) and pump-motor (charge) modes, plus reservoir energy capacity.

## Model Equations

**Generation (discharge):**
```
P_gen = eta_turbine * eta_generator * rho * g * Q * H / 1000   [kW]
```

**Pumping (charge):**
```
P_pump = rho * g * Q * H / (eta_pump * eta_motor * 1000)        [kW]
```

**Round-trip efficiency:**
```
RTE = eta_turbine * eta_generator * eta_pump * eta_motor
```

**Stored energy capacity:**
```
E = rho * g * V_reservoir * H / 3.6e9   [GWh]
```

## Default Parameters
| Parameter | Value | Unit |
|-----------|-------|------|
| Head (H) | 300 | m |
| Design flow rate | 50 | m³/s |
| eta_turbine | 0.90 | — |
| eta_pump | 0.88 | — |
| eta_generator | 0.97 | — |
| eta_motor | 0.97 | — |
| Reservoir volume | 5×10⁶ | m³ |

## Inputs
| Name | Unit | Range | Description |
|------|------|-------|-------------|
| mode | — | generate / pump | Operating mode |
| flow_rate | m³/s | [1, 500] | Volumetric flow rate |
| head | m | [10, 1000] | Net hydraulic head |

## Outputs
| Name | Unit | Description |
|------|------|-------------|
| power_kw | kW | Electrical output (generate) or input (pump) |
| efficiency | — | One-way efficiency for given mode |
| energy_capacity_gwh | GWh | Stored energy for default reservoir at given head |
| round_trip_eta | — | Overall round-trip efficiency |

## Validation Checks
- P_pump > P_gen for same Q and H (losses)
- Round-trip eta ~0.75–0.80 (typical PHS range)
- Power linear in Q and H

## Design-Point Results (Q=50 m³/s, H=300 m)
- Generation power: ~132 MW
- Pumping power: ~172 MW
- Round-trip efficiency: ~0.755
- Energy capacity: ~4.1 GWh

## Sources
1. Rehman, S., Al-Hadhrami, L.M., Alam, M.M. (2015). Pumped hydro energy storage system:
   A technological review. *Renewable and Sustainable Energy Reviews*, 44, 586–598.

## Limitations
- No hydraulic friction losses (head treated as net head)
- No cavitation or part-load efficiency correction
- Fixed reservoir geometry (no drawdown effects)
- No ramp-rate or startup constraints
