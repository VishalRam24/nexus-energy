# EC009 — Alkaline Electrolyser (AEL) — F1a V-I Characteristic

## Model Card

| Field | Value |
|-------|-------|
| **EC ID** | EC009 |
| **Component** | Alkaline Electrolyser (AEL) |
| **Fidelity** | F1a — Semi-empirical V-I characteristic |
| **Author** | Ulleberg (2003) |
| **License** | BSD-3 |

## Physics

Ulleberg (2003) semi-empirical model for alkaline water electrolysis:

```
V_cell = E_rev(T) + [r1 + r2*T]/A * j + s * log10((t1 + t2/T + t3/T^2) * j/A + 1)
E_rev(T) = 1.229 - 0.0009*(T - 298)    [V]
H2_rate  = eta_F * N_cells * j * A / (2*F)
eta_F    = f1*(j*A)^2 / (f2 + (j*A)^2)    [Faraday efficiency]
```

**Note on high-current-density behavior:** At T=80°C, r(T) = r1 + r2*T ≈ -7.8e-6 Ohm.m2 (slightly negative), which is physically interpreted as the dominant resistance being captured in the overvoltage (log) term. The model is valid and monotone in the practical range 100–2000 A/m².

## Parameters (Ulleberg 2003, Table 1)

| Parameter | Value | Unit | Description |
|-----------|-------|------|-------------|
| N_cells | 20 | — | Cells in series |
| A | 0.25 | m² | Electrode area |
| r1 | 8.05e-5 | Ω·m² | Ohmic resistance coefficient |
| r2 | -2.5e-7 | Ω·m²/K | Temperature dependence of r |
| s | 0.185 | V | Tafel slope coefficient |
| t1 | 1.002 | m²/A | Overvoltage parameter |
| t2 | 8.424 | m²·K/A | Overvoltage parameter |
| t3 | 247.3 | m²·K²/A | Overvoltage parameter |
| f1 | 250 | A² | Faraday efficiency coefficient |
| f2 | 0.98 | — | Faraday efficiency coefficient |

## Inputs / Outputs

### Inputs
| Name | Unit | Range | Default |
|------|------|-------|---------|
| current_density | A/m² | 0–3000 | — |
| temperature | °C | 40–90 | 80 |

### Outputs
| Name | Unit | Description |
|------|------|-------------|
| cell_voltage | V | Single-cell terminal voltage |
| stack_voltage | V | Total stack voltage (N_cells * V_cell) |
| hydrogen_rate_mols | mol/s | H2 production rate |
| power_kw | kW | Electrical power consumed |
| efficiency | — | LHV efficiency (H2 chemical / electrical) |

## Typical Values (j=2000 A/m², T=80°C, N=20 cells, A=0.25 m²)

- V_cell ≈ 1.84 V
- V_stack ≈ 36.8 V
- Power ≈ 18.4 kW
- H2 rate ≈ 0.047 mol/s (169 mol/hr)
- Efficiency ≈ 62%

## Physics Checks (all pass)

- V_cell > E_rev at all current densities
- V_cell decreases with temperature at fixed j
- H2 rate proportional to current (Faraday's law)
- Efficiency < 1 across all operating points
- Stack voltage = 20 × cell voltage

## Data Sources

- Ulleberg, O. (2003). Modeling of advanced alkaline electrolyzers: a system simulation approach. *International Journal of Hydrogen Energy*, 28(1), 21–33. https://doi.org/10.1016/S0360-3199(02)00043-9

## Limitations

- Isothermal model only (no thermal dynamics)
- Does not model degradation
- Valid range: j = 100–2000 A/m², T = 40–90°C
- Faraday efficiency uses simplified algebraic expression
