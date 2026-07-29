# EC061 — Unglazed Solar Collector (Pool Heating) — F1b: IAM + Wind + Sky Radiation

## Model Summary
Extends F1a by adding ASHRAE b0 IAM, wind-speed correction on heat loss coefficient, and sky radiation loss. Unglazed collectors (EPDM rubber mats) are the dominant low-cost technology for pool heating.

## Physics
- **IAM**: ASHRAE 93 b0 model: IAM(θ) = 1 − b0 × (1/cos θ − 1). For smooth EPDM b0 ≈ 0.07 (lower than glazed collectors)
- **Wind heat loss**: U_L(v) = U_L0 + U_wind × v_wind. Critical for unglazed (no protective cover)
- **Sky radiation**: Q_sky = ε × σ × A × (T_col⁴ − T_sky⁴). Significant at night/cloudy conditions
- **Net heat**: Q_u = A × F_R × [IAM × τα × G − U_L(v) × (T_in − T_amb)] − F_R × Q_sky

## Key Physics Notes
- Unglazed: U_L ~15–30 W/m²K (vs 3–5 for glazed). Pool heating only viable for small ΔT
- Wind is the dominant loss mechanism — field-measured k_wind ≈ 3 W/(m²K·m/s)
- b0 ≈ 0.05–0.10 for smooth polymer surfaces (much smaller than glazed collectors)

## Reference Parameters
10 m², F_R = 0.92, τα = 0.86, U_L0 = 15 W/m²K, U_wind = 3 W/(m²K·m/s), b0 = 0.07

## References
- Duffie & Beckman (2013), Solar Engineering of Thermal Processes, Ch. 6, 10
- ASHRAE Standard 93 (2010)
- ISO 9806:2017 — Solar thermal collectors test methods
- Martinopoulos et al. (2010), Solar Energy 84(1), 117–127
