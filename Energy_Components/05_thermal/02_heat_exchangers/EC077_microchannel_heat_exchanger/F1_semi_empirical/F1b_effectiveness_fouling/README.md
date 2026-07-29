# EC077 — Microchannel Heat Exchanger — F1b: ε-NTU + Fouling + Part-Load LMTD

## Model Summary
Extends F1a (clean ε-NTU) with fouling resistance correction and a part-load LMTD correction factor for cross-flow microchannels. Microchannel HX have very high U_clean (~2500 W/m²K) but are particularly sensitive to fouling due to small hydraulic diameter.

## Physics
- **Fouling**: 1/U_f = 1/U_clean + Rf_hot + Rf_cold (TEMA resistances)
- **Part-load LMTD**: F(PLR) = f_a + f_b × (PLR − 1). Accounts for cross-flow deviation from counter-flow and maldistribution at reduced flow
- **ε-NTU**: Counter-flow formula with NTU = U_f × A × F / C_min
- **Effectiveness reduction**: (ε_clean − ε_fouled) / ε_clean

## Key Physics Notes
- Microchannel Dh ~ 0.5 mm → U_clean 10× higher than shell-and-tube
- Fouling impact is proportionally larger: Rf=0.0002 m²K/W reduces U by ~30% vs ~5% for large tubes
- Cross-flow correction F ≈ 0.92 at full load; drops to ~0.87 at 50% PLR

## Reference Parameters
U_clean = 2500 W/m²K, A = 2 m², cross-flow aluminum HX

## References
- Incropera & DeWitt (2006), Fundamentals of Heat and Mass Transfer, Ch. 11
- TEMA Standards, 10th ed. — Fouling resistance tables
- Kandlikar & Shah (2012), J. Heat Transfer 120(4)
- Kakaç & Liu (2002), Heat Exchangers: Selection, Rating, and Thermal Design
