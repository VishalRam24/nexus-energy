# EC168 -- MPPT Controller -- F1b P&O Algorithm Loss Model

## Overview
Extends F1a (empirical tracking efficiency curve) with a physics-based P&O algorithm model
that separately accounts for three loss mechanisms: MPP oscillation, dynamic tracking lag,
and DC-DC converter losses.

## Model Equations

**Oscillation loss (P&O steady-state perturbation):**
```
P_osc = 0.5 * |dP/dV|_mpp * V_step^2 / V_mpp
eta_static = 1 - P_osc / P_mpp
```

**Dynamic tracking loss (irradiance transients):**
```
eta_dynamic = 1 - |dG/dt| * T_mppt / G
```

**Converter loss:**
```
eta_converter = constant (from DC-DC converter model)
```

**Total:**
```
eta_total = eta_static * eta_dynamic * eta_converter
P_out = P_mpp * eta_total
```

## Default Parameters
| Parameter | Value | Unit | Description |
|-----------|-------|------|-------------|
| V_step | 0.5 | V | P&O perturbation step |
| T_mppt | 50 | ms | MPPT update period |
| eta_static | 99.5 | % | Nominal static tracking efficiency |
| eta_converter | 97 | % | DC-DC converter efficiency |
| dP/dV_mpp | -50 | W/V | P-V curve slope at MPP |

## Sources
1. Hohm, D.P. & Ropp, M.E. (2003). Prog. Photovoltaics, 11, 47-62.
2. Femia, N. et al. (2005). IEEE Trans. Power Electron., 20(4), 963-973.

## Limitations
- P&O algorithm only (no incremental conductance or other algorithms)
- Linear scaling of dP/dV with irradiance (simplified)
- No partial shading or multi-peak tracking
- Converter efficiency is constant (not load-dependent)
