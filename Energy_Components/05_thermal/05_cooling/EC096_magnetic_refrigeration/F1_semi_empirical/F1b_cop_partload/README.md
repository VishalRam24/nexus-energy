# EC096 — Magnetic Refrigeration — F1b: COP vs Temperature + Part-Load Penalty

## Model Summary
Semi-empirical AMR (Active Magnetic Regenerator) model combining Carnot-based efficiency with AMR system losses, hot-side temperature correction, and quadratic part-load penalty. Near-room-temperature systems use gadolinium or La(Fe,Si)₁₃-based alloys.

## Physics
- **Carnot COP**: T_cold_K / (T_hot_K − T_cold_K)
- **AMR efficiency**: η_AMR accounts for magnet work, regenerator thermal mass, and cycle irreversibilities (~25–35% of Carnot)
- **Temperature correction**: f_T = 1 − k_T × (T_hot − T_design). Harder to reject heat at higher temperatures
- **Part-load penalty**: f_PLR = p1 + p2×PLR + p3×PLR². AMR operates best at rated oscillation frequency

## Key Physics Notes
- Magnetic refrigeration has no compressor; work is from rotating/cycling magnetic field
- Part-load is achieved by reducing field rotation speed → MCE not maximized → COP penalty
- Current state-of-art: COP ~ 1.5–3.5 for 20 K span (approaching vapor compression)
- η_AMR ~ 0.30 (30% of Carnot) is a conservative current-technology estimate

## Reference Parameters
10 kW cooling, T_hot_design = 35°C, T_cold_design = 15°C, η_AMR = 0.30

## References
- Kitanovski et al. (2015), Magnetocaloric Energy Conversion, Springer
- Yu et al. (2010), Int. J. Refrigeration 33(6), 1029–1060
- Aprea et al. (2015), Int. J. Refrigeration 52, 98–108
