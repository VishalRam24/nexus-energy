# EC156 Geothermal Heat Pump (GHP) — F1b Ground Thermal Degradation

## Model Summary
Extends F1a COP map with ground thermal saturation during heating season, brine fouling from TDS, part-load COP correction, and reservoir-style temperature decline.

## Key Physics
- **Ground thermal saturation**: `ΔT_ground = Q * R_th * (1 - exp(-t/τ))`, τ=800h
- **Brine fouling**: `R_f = k_f * (TDS/TDS_ref)^0.5 * t^0.3`, f_foul = 1/(1+R_f*U)
- **Part-load COP**: Interpolated from empirical curve; cycling losses below 30% PLR
- **Condenser T sensitivity**: Carnot fraction approach from F1a with time-varying T_source

## References
- Staffell, I. et al. (2012). Energy Environ. Sci., 5, 9291-9306.
- ASHRAE (2011). Geothermal Heating and Cooling Design Guide.
- Kavanaugh, S.P. & Rafferty, K. (2014). Geothermal Heating and Cooling. ASHRAE Press.
- Yang, H. et al. (2010). Applied Energy, 87, 16-27.
