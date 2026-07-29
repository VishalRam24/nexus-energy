# EC057 — Stirling Dish CSP — F1b: Optical + Receiver Thermal Loss Model

## Model Summary
Semi-empirical model combining optical efficiency, physics-based receiver cavity heat losses, and Stirling engine performance. Extends F1a (optical-only) with:
- Cavity convective + radiative + conductive losses
- Stirling efficiency from modified Carnot + part-load correction

## Physics
- **Optical**: Q_abs = DNI × A_dish × η_opt × IAM(θ); IAM = cos(θ) for 2-axis tracking
- **Receiver losses**: Q_loss = A_rec × [h_cav × ΔT + U_cond × ΔT + ε·σ·(T_rec⁴ − T_sky⁴)]
- **Stirling efficiency**: η = η_int × (1 − T_sink/T_hot) × f_PLR × η_alt
- **Net output**: P_elec = (Q_abs − Q_loss) × η_Stirling

## Inputs
| Input | Unit | Range |
|-------|------|-------|
| DNI | W/m² | 0–1100 |
| Incidence angle θ | deg | 0–10 |
| Receiver temperature | °C | 400–800 |
| Ambient temperature | °C | -10 to 45 |
| PLR | — | 0.3–1.0 |

## Outputs
| Output | Unit |
|--------|------|
| power_output_kw | kW |
| Q_absorbed_kw | kW |
| Q_receiver_loss_kw | kW |
| Q_net_thermal_kw | kW |
| eta_stirling | — |
| overall_efficiency | — |
| iam_factor | — |

## Reference Parameters
25 kW dish (SunCatcher/EuroDish class): A_dish = 91 m², η_opt = 0.88, η_int = 0.40, η_alt = 0.92

## References
- Mancini et al. (2003), J. Sol. Energy Eng. 125(2), 135–151
- Nepveu et al. (2009), Sol. Energy 83(1), 81–89
- Stine & Diver (1994), SAND93-7026, Sandia NL
