# EC139 — Salinity Gradient Blue Energy (PRO) — F1b Membrane Resistance Model

## Model Description
Extends F1a (bulk Gibbs energy) with a **solution-diffusion membrane transport model**
including internal and external concentration polarization (ICP/ECP) and
temperature-dependent diffusivity.

**Energy basis:** per m³ freshwater permeated (Yip & Elimelech 2012 Phase 7 convention).

## Physics Added over F1a
| Feature | Equation | Reference |
|---------|----------|-----------|
| Water flux | J_w = A_w × (ΔΠ_eff − ΔP) | Achilli & Childress (2010) |
| ICP (dilutive) | C_fs = C_fw × exp(J_w × S / D) | Loeb et al. (1997) |
| ECP (dilutive) | C_ds = C_sw × exp(−J_w / k_d) | Yip & Elimelech (2012) |
| T-dependent D | D(T) = D_ref × (T/T_ref) × (μ_ref/μ(T)) | Stokes-Einstein |
| Pump parasitic | P_pump = Q_fw × ΔP / η_pump × (1 − η_px) | Straub et al. (2016) |

## Inputs
| Parameter | Unit | Range | Default |
|-----------|------|-------|---------|
| C_sw | g/L | 25–40 | 35.0 |
| C_fw | g/L | 0.1–2.0 | 0.5 |
| dP_bar | bar | 0–30 | 12.0 |
| T_degC | °C | 0–40 | 25.0 |

## Outputs
| Parameter | Unit | Description |
|-----------|------|-------------|
| J_w_m_s | m/s | Membrane water flux |
| dPi_eff_bar | bar | Effective osmotic pressure after CP |
| power_density_W_m2 | W/m² | Gross power density |
| net_energy_kwh_per_m3 | kWh/m³_fw | Net energy per m³ freshwater (Phase 7) |
| power_kw | kW | Net electrical power |
| cp_factor_ICP | — | C_fs/C_fw (>1, dilutive ICP) |
| cp_factor_ECP | — | C_ds/C_sw (<1, dilutive ECP) |

## Typical Results
- Seawater (35 g/L) vs river water (0.5 g/L), T=25°C, dP_opt ≈ 11–13 bar:
  - J_w ≈ 5–15 L/(m²·h)
  - Power density ≈ 1–5 W/m²
  - Net energy ≈ 0.15–0.35 kWh/m³ freshwater

## References
- Yip, N.Y. & Elimelech, M. (2012). *Environ. Sci. Technol.* 46, 5230–5239.
- Achilli, A. & Childress, A.E. (2010). *Desalination* 261, 205–211.
- Loeb, S. et al. (1997). *J. Membr. Sci.* 129, 243–249.
- Straub, A.P. et al. (2016). *Nature Energy* 1, 16090.

## Limitations
- Steady-state model; no fouling/scaling dynamics
- Uniform membrane properties assumed (no spatial variation)
- NaCl-only electrolyte (real seawater is multi-ion)
- B-parameter (salt leakage) included in flux but not in osmotic pressure correction
