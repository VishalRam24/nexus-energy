# EC010 — Solid Oxide Electrolyser (SOEC) — F1a V-I Characteristic

## Model Card

| Field | Value |
|-------|-------|
| **EC ID** | EC010 |
| **Component** | Solid Oxide Electrolyser (SOEC) |
| **Fidelity** | F1a — Semi-empirical V-I characteristic |
| **Author** | Ni et al. (2007) |
| **License** | BSD-3 |

## Physics

ASR-based semi-empirical model for high-temperature steam electrolysis:

```
V_cell = E_rev(T) + j * ASR(T)
E_rev(T) = 1.253 - 0.00024*(T - 298)       [V]
ASR(T)   = ASR_ref * exp(E_act/R * (1/T - 1/T_ref))   [Ω·cm²]  Arrhenius
H2_rate  = N_cells * j * A / (2*F)          [mol/s]   (100% Faraday efficiency)
```

**Thermo-neutral voltage:** V_tn ≈ 1.285 V (at 800°C). If V_cell < V_tn, operation is endothermic (needs external heat); if V_cell > V_tn, operation is exothermic.

## Parameters

| Parameter | Value | Unit | Description |
|-----------|-------|------|-------------|
| N_cells | 30 | — | Cells in series |
| A | 100 | cm² | Electrode area |
| T_nominal | 1073 | K | Operating temperature (800°C) |
| ASR_ref | 0.3 | Ω·cm² | ASR at T_ref |
| E_act | 60,000 | J/mol | Activation energy |
| T_ref | 1073 | K | Reference temperature |

## Inputs / Outputs

### Inputs
| Name | Unit | Range | Default |
|------|------|-------|---------|
| current_density | A/cm² | 0–2.0 | — |
| temperature | °C | 600–900 | 800 |

### Outputs
| Name | Unit | Description |
|------|------|-------------|
| cell_voltage | V | Single-cell terminal voltage |
| stack_voltage | V | Total stack voltage |
| hydrogen_rate_mols | mol/s | H2 production rate |
| power_kw | kW | Electrical power consumed |
| efficiency | — | LHV efficiency |
| asr | Ω·cm² | Area-specific resistance at given T |

## Typical Values (j=1.0 A/cm², T=800°C)

- E_rev ≈ 1.067 V
- ASR = 0.300 Ω·cm²
- V_cell ≈ 1.367 V (above V_tn, exothermic)
- V_stack ≈ 41.0 V
- H2 rate ≈ 0.016 mol/s
- Efficiency ≈ 78%

## Physics Checks (all pass)

- V_cell ≥ E_rev at all current densities
- V_cell strictly increases with j
- ASR strictly decreases with temperature (Arrhenius)
- H2 rate exactly proportional to current (100% Faraday efficiency)
- Stack voltage = 30 × cell voltage

## Data Sources

- Ni, M., Leung, M. K. H., & Leung, D. Y. C. (2007). Technological development of hydrogen production by solid oxide electrolyzer cell (SOEC). *Chemical Engineering & Technology*, 29(6), 636–642. https://doi.org/10.1002/ceat.200600330

## Limitations

- Assumes 100% Faraday efficiency (valid at SOEC operating temperatures)
- Isothermal — no thermal dynamics or heat integration
- No gas concentration polarization term
- Valid range: j = 0.1–2.0 A/cm², T = 600–900°C
- Does not model degradation (Cr poisoning, delamination)
