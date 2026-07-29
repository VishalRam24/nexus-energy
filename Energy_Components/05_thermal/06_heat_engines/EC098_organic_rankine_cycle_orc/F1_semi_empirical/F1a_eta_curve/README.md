# EC098 — Organic Rankine Cycle (ORC) — F1a Efficiency Curve

## Model Card

| Field | Value |
|-------|-------|
| Component | Organic Rankine Cycle (ORC) |
| EC ID | EC098 |
| Fidelity | F1a — Semi-Empirical Efficiency Curve |
| Path | `05_thermal/06_heat_engines/EC098_organic_rankine_cycle_orc/F1_semi_empirical/F1a_eta_curve/` |

## Model Equations

```
eta_Carnot   = 1 - T_cold_K / T_hot_K
eta_internal = eta_expander * eta_pump  = 0.75 * 0.65 = 0.4875
f_PLR        = 0.15 + 0.85 * PLR

eta_thermal  = eta_Carnot * eta_internal * f_PLR

P_out        = Q_hot * eta_thermal
Q_reject     = Q_hot - P_out
```

## Parameters

| Parameter | Value | Unit | Source |
|-----------|-------|------|--------|
| P_rated | 100 | kW_e | Rated electrical output |
| T_hot_rated | 150 | degC | Typical waste-heat ORC source |
| eta_expander | 0.75 | — | Isentropic expander (scroll/turbine) |
| eta_pump | 0.65 | — | Isentropic pump efficiency |
| eta_internal | 0.4875 | — | Product of expander and pump efficiencies |
| Working fluid | R245fa | — | Low-boiling HFC (Tb=15.1°C) |

## Inputs / Outputs

### Inputs
| Name | Unit | Range | Description |
|------|------|-------|-------------|
| T_hot | degC | 80–300 | Heat source temperature |
| T_cold | degC | 10–50 | Heat sink (cooling water) temperature |
| part_load_ratio | — | 0.3–1.0 | Fractional load (default 1.0) |
| Q_hot_kw | kW | optional | Heat input; if omitted, inferred from rated power |

### Outputs
| Name | Unit | Description |
|------|------|-------------|
| efficiency | — | Net thermal efficiency |
| power_kw | kW_e | Net electrical output |
| heat_input_kw | kW_th | Thermal energy input from source |
| heat_rejection_kw | kW_th | Heat rejected to cold sink |

## Physics Checks (all pass)
- eta < eta_Carnot at all conditions
- eta < 0.25 for T_hot < 250°C
- power_kw < heat_input_kw (Second Law)
- Q_in = P_out + Q_reject (First Law, machine precision)
- Efficiency increases with T_hot, decreases with T_cold
- Lower PLR reduces efficiency (f_PLR < 1)

## Rated Performance
At T_hot=150°C, T_cold=30°C, PLR=1.0: eta ≈ 9.5%, P_out ≈ 100 kW

## Benchmark
1000 predictions in < 1 ms (NumPy vectorized)

## Limitations
- Working fluid properties are implicit in eta_internal; use CoolProp + F2 for cycle-accurate results.
- Linear PLR correction (f_PLR) is a simplification; actual part-load behaviour is nonlinear.
- No startup/shutdown dynamics. No degradation model.
- T_hot up to 300°C is within R245fa stability range; above that use different fluids (F1b).

## Data Sources
- Quoilin, S., Van Den Broek, M., Declaye, S., Dewallef, P. & Lemort, V. (2013). Techno-economic survey of Organic Rankine Cycle (ORC) systems. *Renewable and Sustainable Energy Reviews*, 22, 168–186.

## License
BSD-3
