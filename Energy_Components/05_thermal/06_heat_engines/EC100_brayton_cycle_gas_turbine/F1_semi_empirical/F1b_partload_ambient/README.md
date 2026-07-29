# EC100 — Brayton Cycle Gas Turbine (Simple Cycle) — F1b: Part-Load + Ambient

## Model Summary
Semi-empirical simple-cycle GT model combining a quadratic part-load efficiency correction with ISO 2314 ambient temperature and pressure corrections. Based on F-class GT data (GE 7F.04 class, 185 MW ISO).

## Physics
- **Part-load**: f_PLR(PLR) = a + b×PLR + c×PLR². GTs have steep efficiency drop below 60% load
- **Ambient temperature**: η_corr = η_ISO × √(T_ISO/T_amb); P_corr = P_ISO × (P/P_ISO) × √(T_ISO/T_amb)
- **Exhaust temperature**: T_exh = T_exh_ISO + k_PLR × (1 − PLR). Hotter exhaust at part-load (less mass flow, similar heat release)
- **Heat rate**: HR = 3600 / η [kJ/kWh]

## Key Physics Notes
- Simple cycle GTs are very sensitive to ambient temperature: −1°C → +0.1% power, +0.04% η
- PLR_min ≈ 40% for F-class GT (combustion stability limit)
- Exhaust temperature 540–600°C at full load — valuable for HRSG in combined cycle

## Reference Parameters
185 MW, η_rated = 38.5%, ISO 15°C/101.325 kPa, T_exh ≈ 580°C at full load

## References
- Walsh & Fletcher (2004), Gas Turbine Performance, 2nd ed., Blackwell
- ISO 2314:2009 — Gas turbines: Acceptance tests
- Horlock (2003), Advanced Gas Turbine Cycles, Elsevier
- GE Power F-class GT data sheets
