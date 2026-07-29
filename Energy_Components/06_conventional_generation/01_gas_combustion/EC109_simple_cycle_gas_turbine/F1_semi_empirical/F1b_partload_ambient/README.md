# EC109 -- Simple Cycle Gas Turbine -- F1b Part-Load + Ambient Correction

## Overview
Extends F1a by adding ISO-standard ambient pressure and temperature corrections to both power output and efficiency, plus an exhaust temperature model. This enables site-specific performance estimation under non-ISO conditions.

## Model Equations
```
Part-load efficiency correction:
    f_PLR(PLR) = a + b*PLR + c*PLR^2
    (a=0.1, b=1.3, c=-0.4 -> peak near PLR~0.85)

ISO ambient power correction:
    P_corrected = P_rated * PLR * (P_amb / P_ref) * sqrt(T_ref / T_amb)

Efficiency with ambient correction:
    eta(PLR, T_amb) = eta_rated * f_PLR(PLR) * sqrt(T_ref / T_amb)

Exhaust temperature:
    T_exhaust = T_exh_rated + dT_partload * (1 - PLR)

Fuel flow:
    fuel_flow = P_out / (eta * LHV)  [kg/s]

Heat rate:
    HR = 3600 / eta  [kJ/kWh]
```

## Inputs
| Name | Unit | Range | Default | Description |
|------|------|-------|---------|-------------|
| PLR | - | [0.3, 1.0] | - | Part-load ratio |
| T_ambient | K | [243, 323] | 288.15 | Ambient temperature |
| P_ambient | kPa | [80, 110] | 101.325 | Ambient pressure |
| fuel_lhv | MJ/kg | [40, 55] | 50.0 | Fuel lower heating value |

## Outputs
| Name | Unit | Description |
|------|------|-------------|
| efficiency | - | Net LHV electrical efficiency |
| power_output_kw | kW | Electrical power output |
| fuel_flow_kg_s | kg/s | Fuel mass flow rate |
| exhaust_temp_K | K | Exhaust gas temperature |
| heat_rate_kj_kwh | kJ/kWh | Heat rate |

## Key Physics
- Higher ambient temperature reduces air density, lowering mass flow and power output
- Higher ambient pressure increases air density and power output
- Part-load efficiency drops due to off-design compressor/turbine matching
- Exhaust temperature slightly increases at part load (reduced expansion ratio)

## Sources
1. Walsh & Fletcher (2004), Gas Turbine Performance, 2nd ed., Blackwell Science.
2. ISO 2314:2009 Gas turbines -- Acceptance tests.

## Limitations
- Steady-state only; no transient/startup dynamics
- Single-shaft GT assumed; multi-shaft designs may differ
- No humidity correction (can add ~1-2% variation)
- Exhaust temperature model is simplified (linear with PLR)
