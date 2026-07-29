# EC098 -- Organic Rankine Cycle (ORC) -- F1b Part-Load + Condenser Ambient Effect

## Overview
Extends F1a by adding a quadratic part-load correction and condenser temperature sensitivity. ORC efficiency is very sensitive to condenser temperature because the Carnot efficiency is low, making small absolute changes in T_cold cause large relative efficiency changes.

## Model Equations
```
Carnot efficiency:
    eta_carnot = 1 - T_cold_K / T_hot_K

Design efficiency:
    eta_design = eta_carnot * eta_internal

Part-load correction:
    f_PLR(PLR) = a + b*PLR + c*PLR^2

Condenser temperature correction:
    f_T = 1 - k_T * (T_cond - T_cond_design)

Combined:
    eta = eta_design * f_PLR * f_T   (capped at eta_carnot)

Power output:
    P_out = Q_hot * eta

Heat rejection:
    Q_reject = Q_hot - P_out
```

## Inputs
| Name | Unit | Range | Default |
|------|------|-------|---------|
| T_heat_source | degC | [80, 300] | 150.0 |
| T_condenser | degC | [15, 55] | 30.0 |
| PLR | - | [0.3, 1.0] | 1.0 |
| heat_input_kw | kW | [50, 2000] | auto |

## Outputs
| Name | Unit | Description |
|------|------|-------------|
| efficiency | - | Net ORC thermal efficiency |
| power_output_kw | kW | Electrical output |
| heat_rejection_kw | kW | Heat rejected to condenser |
| specific_work_kj_kg | kJ/kg | Approximate specific work |

## Key Physics
- ORC efficiency very sensitive to condenser temperature (~1.2% relative loss per K)
- At 50% part-load, efficiency drops to ~85% of rated (pump/expander mismatch)
- Efficiency always bounded by Carnot limit
- Lower heat source temperatures yield lower efficiency (thermodynamic limit)

## Sources
1. Quoilin et al. (2013), Techno-economic survey of ORC systems, RSER 22, 168-186.
2. Manente et al. (2013), Off-design performance of ORC power plants.
3. Lecompte et al. (2015), Review of ORC for low-grade waste heat recovery.

## Limitations
- Steady-state only
- Working fluid (R245fa) properties not explicitly modelled
- No pinch-point analysis for evaporator/condenser
- Specific work is approximate (enthalpy estimated from temperature difference)
