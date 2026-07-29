# EC017 Hydrogen Purifier (PSA) — F1b Temperature-Pressure

## F1b Additions over F1a
- Temperature-dependent recovery: eta_T = eta_P * (1 + k_T*(T - T_ref)); k_T < 0
- Temperature-dependent specific energy: W = W_nom * (P_ref/P)^0.15 * (T/T_ref)^0.5
- The (P_ref/P)^0.15 exponent is per Sircar & Golden (2000)

## References
- Sircar & Golden (2000). Sep. Sci. Technol. 35(5), 667-687.
- Yang (1987). Gas Separation by Adsorption Processes.
- Ruthven (1984). Principles of Adsorption.
