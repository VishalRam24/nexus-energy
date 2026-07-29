# EC003 — Alkaline Fuel Cell (AFC) — F1b Polarization-Thermal Model

## Summary
Temperature-dependent polarization curve for AFC using KOH liquid electrolyte.
Extends F1a (isothermal) by coupling the KOH conductivity, exchange-current
density, and activation losses to operating temperature.

## Key Physics Added Over F1a
| Feature | Model |
|---------|-------|
| KOH conductivity | Gilliam et al. (2007) empirical fit: σ = f(T, c_KOH) |
| Exchange current density | Arrhenius: i₀(T) = i₀_ref × exp(−Eₐ/R × (1/T − 1/T_ref)) |
| Activation overpotential | Butler-Volmer arcsinh form, T-dependent |
| Ohmic loss | V_ohm = j × L / σ_KOH(T) |
| Concentration loss | −B × ln(1 − j/j_L) |
| Heat generation | Q = j × (E_tn − V_cell) |

## Inputs
| Name | Unit | Range | Description |
|------|------|-------|-------------|
| current_density | A/cm2 | 0–1.0 | Operating current density |
| temperature | K | 333–363 | Stack temperature (60–90 C) |
| pressure_h2 | atm | 0.5–3 | H2 partial pressure (optional) |
| pressure_o2 | atm | 0.1–1 | O2 partial pressure (optional) |

## Outputs
| Name | Unit | Description |
|------|------|-------------|
| cell_voltage_V | V | Net cell voltage |
| power_density_W_cm2 | W/cm2 | Power density |
| efficiency | - | Voltage efficiency (V/E_tn) |
| heat_generation_W_cm2 | W/cm2 | Waste heat per unit area |
| electrolyte_resistance_ohm_cm2 | Ω·cm2 | KOH ASR |
| koh_conductivity_S_cm | S/cm | KOH conductivity at operating T |

## References
- Gilliam R.J. et al. (2007). A review of specific conductivities of potassium hydroxide solutions.
  *Int. J. Hydrogen Energy*, 32(3), 359–364.
- Appleby A.J. & Foulkes F.R. (1989). *Fuel Cell Handbook*. Van Nostrand Reinhold.
- Larminie J. & Dicks A. (2003). *Fuel Cell Systems Explained*, 2nd Ed. Wiley.

## Limitations
- KOH regression valid for 2–18 mol/L and 273–373 K; extrapolation not recommended.
- CO2 poisoning of KOH not modelled (F2 handles this).
- Liquid water management not included.
