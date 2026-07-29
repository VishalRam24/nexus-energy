# EC005 — Molten Carbonate Fuel Cell (MCFC) — F1b Polarization-Thermal Model

## Summary
Temperature-dependent polarization for MCFC with molten Li₂CO₃/K₂CO₃ electrolyte.
Extends F1a by including temperature-dependent carbonate conductivity, CO2-corrected
Nernst potential, and Arrhenius kinetics for both anode and cathode.

## Key Physics Added Over F1a
| Feature | Model |
|---------|-------|
| Carbonate conductivity | Arrhenius: σ(T) = A_mc × exp(−E_act_mc/(RT)) |
| CO2 Nernst correction | E = E₀(T) + RT/(2F) × ln(pH₂ × pCO₂_cat × √pO₂ / (pH₂O × pCO₂_an)) |
| Anode exchange current | Arrhenius i₀_a(T) |
| Cathode exchange current | Arrhenius i₀_c(T) (ORR in carbonate) |
| Ohmic loss | V_ohm = j × t_mc / σ(T) |
| Heat generation | Q = j × (E_tn − V_cell) |

## References
- Uchida I. et al. (1983). Electrochim. Acta, 28(10), 1423–1431.
- Lu S.T. & Selman J.R. (1984). J. Electrochem. Soc., 131(12), 2827–2833.
- Yuh C. & Selman J.R. (1991). J. Electrochem. Soc., 138(12), 3649–3655.

## Limitations
- Internal reforming not modelled; fuel composition assumed fixed.
- CO2 recycling system not simulated.
- Valid 873–973 K; extrapolation outside this range unreliable.
