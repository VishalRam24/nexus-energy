# EC102 — Kalina Cycle — F1b: Part-Load + Condenser T + Ammonia Fraction

## Model Summary
Semi-empirical Kalina cycle model extending F1a with ammonia composition correction, condenser temperature sensitivity, and part-load penalty. The key Kalina advantage over ORC is the composition-tunable boiling/condensing curve, modeled via the f_composition factor.

## Physics
- **Carnot base**: η_Carnot = 1 − T_cond_K / T_hot_K
- **NH3 fraction correction**: f_x = 1 + k_x × (x_NH3 − x_design). Composition can be tuned for different heat source temperatures
- **Condenser sensitivity**: f_T = 1 − k_T × (T_cond − T_design). Ammonia has very high dp/dT, making condenser T critical
- **Part-load**: f_PLR = p1 + p2×PLR + p3×PLR²
- **Net**: η = η_Carnot × η_int × f_x × f_T × f_PLR

## Key Physics Notes
- Kalina achieves η_internal ~ 50% of Carnot (vs ~35% for ORC) due to composition matching
- Condenser k_T = 0.02 1/K means 15K rise → ~30% efficiency drop (important for hot-summer sites)
- Optimal x_NH3 shifts with T_hot: ~0.7 for 100°C, ~0.85 for 150°C sources
- Kalina cycles are more complex to operate than ORC but outperform for T_hot < 200°C

## Reference Parameters
100 kW, T_hot = 150°C, T_cond = 32°C, x_NH3 = 0.85, η_int = 0.50

## References
- Kalina (1984), US Patent 4,346,561
- Bombarda et al. (2010), Appl. Thermal Eng. 30(2), 212–219
- Lolos & Rogdakis (2009), Energy 34(4), 457–464
- Leibowitz et al. (1997), Modern Power Systems, June 1997
