# EC004 — Phosphoric Acid Fuel Cell (PAFC) — F1b Polarization-Thermal Model

## Summary
Temperature-dependent polarization curve for PAFC using concentrated H3PO4 electrolyte.
Extends F1a by coupling H3PO4 ionic conductivity, cathode exchange-current density,
and thermoneutral voltage to operating temperature (150–210 C).

## Key Physics Added Over F1a
| Feature | Model |
|---------|-------|
| H3PO4 conductivity | Arrhenius: σ(T) = σ_ref × exp(Eₐ_σ/R × (1/T_ref − 1/T)) |
| Cathode exchange current | Arrhenius i₀(T) with E_act = 70 kJ/mol (ORR in H3PO4) |
| Thermoneutral voltage | E_tn(T) = 1.481 − 0.000126×(T−298) |
| Nernst potential | T-dependent standard potential + Nernst correction |
| Ohmic loss | V_ohm = j × t_acid / σ(T) |
| Heat generation | Q = j × (E_tn(T) − V_cell) |

## Inputs / Outputs
See `data/parameters.json` for parameter definitions and valid ranges.

## References
- Razaq M. et al. (1989). J. Electrochem. Soc., 136(2), 385–390.
- Appleby A.J. & Foulkes F.R. (1989). Fuel Cell Handbook. Van Nostrand.
- Li Q. et al. (2003). Chem. Mater. 15(26), 4896–4915.
- Patel K.K. et al. (2012). Int. J. Hydrogen Energy, 37(3), 2346–2359.

## Limitations
- CO tolerance (~1-2% CO) not modelled; CO effect on i₀ requires F2.
- H3PO4 evaporation / electrolyte management not included.
- Valid for 95–100% H3PO4; dilution effects not captured.
