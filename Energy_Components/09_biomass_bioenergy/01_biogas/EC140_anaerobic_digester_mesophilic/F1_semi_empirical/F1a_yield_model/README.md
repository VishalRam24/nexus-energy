# EC140 — Anaerobic Digester (Mesophilic) — F1a Biogas Yield Model

## Overview
Simplified Buswell/ADM1 biogas yield model for a 500 m³ mesophilic CSTR anaerobic digester. Computes methane yield, biogas production, and energy output as a function of VS loading, hydraulic retention time, and temperature.

## Model Equations
```
methane_yield = Y_max * (1 - exp(-k * HRT)) * f_T(temp)     [m³_CH4/kgVS]

Temperature correction (Arrhenius below 42 degC, linear decay above):
    f_T = exp(-E_a/R * (1/T - 1/T_ref))   for T <= 42 degC
    f_T = f_T(42) * (55-T)/(55-42)        for 42 < T <= 55 degC
    f_T = 0                                 for T > 55 degC

VS_mass      = vs_loading * V_reactor              [kgVS/day]
methane_rate = methane_yield * VS_mass             [m³_CH4/day]
biogas_rate  = methane_rate / methane_fraction     [m³_biogas/day]
energy_output= methane_rate * LHV_methane          [kWh/day]
```

## Inputs
| Name | Unit | Range | Description |
|------|------|-------|-------------|
| vs_loading | kgVS/(m³·day) | [1, 8] | Volatile solids loading rate |
| hrt | days | [5, 40] | Hydraulic retention time |
| temperature | degC | [25, 55] | Reactor operating temperature |

## Outputs
| Name | Unit | Description |
|------|------|-------------|
| methane_yield_m3kgvs | m³_CH4/kgVS | Specific methane yield |
| biogas_rate_m3day | m³/day | Total biogas production rate |
| methane_rate_m3day | m³_CH4/day | Methane production rate |
| energy_output_kwh_day | kWh/day | Equivalent thermal energy from methane |

## Parameters (500 m³ CSTR)
| Parameter | Value | Unit | Source |
|-----------|-------|------|--------|
| V_reactor | 500 | m³ | Design |
| Y_max | 0.35 | m³_CH4/kgVS | ADM1 typical food/manure |
| k | 0.15 | 1/day | ADM1 simplified |
| T_ref | 310 K (37°C) | K | Mesophilic optimum |
| HRT_design | 20 | days | Typical CSTR |
| methane_fraction | 0.60 | - | Typical biogas composition |
| E_a | 55000 | J/mol | Arrhenius activation energy |
| LHV_methane | 9.97 | kWh/m³ | At STP |

## Sources
1. Buswell, A.M. & Mueller, H.F. (1952). Mechanism of methane fermentation. Ind. Eng. Chem., 44(3), 550-552.
2. Batstone, D.J. et al. (2002). Anaerobic Digestion Model No.1 (ADM1). IWA Publishing.

## Physics Checks
- yield < Y_max at all HRT values (asymptotic saturation)
- yield increases monotonically with HRT (more retention = more conversion)
- yield increases with temperature up to 42°C (Arrhenius)
- yield decreases above 42°C (thermal inhibition of methanogenesis)
- biogas_rate / methane_rate = 1/methane_fraction = 1.667 (constant)

## Limitations
- Steady-state model only — no transient startup/shutdown dynamics
- Single substrate (VS) — no substrate-specific yield variation
- No inhibition by ammonia, VFA accumulation, or pH effects
- Temperature inhibition simplified to linear decay (vs. Ratkowsky or exponential models)
- Does not distinguish mesophilic vs. thermophilic microbial communities
