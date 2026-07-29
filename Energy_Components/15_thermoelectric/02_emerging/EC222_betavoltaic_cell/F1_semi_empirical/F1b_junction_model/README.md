# EC222 — Betavoltaic Cell — F1b Junction Electrical Model

**Fidelity:** F1b — Semi-Empirical  
**Sub-branch:** P-N junction model with temperature-dependent Voc, fill factor degradation, decay-coupled Isc

## Model Description

Extends F1a (activity × efficiency = P_out) to a full junction electrical model:

| Parameter | F1a | F1b |
|-----------|-----|-----|
| Power model | P = A*E_beta*eta_cap*eta_conv (constant) | P = Isc*Voc*FF (T and t-dependent) |
| Voc | implicit in eta_conv | explicit: Voc_ref + dVoc/dT*(T-Tref) + n*kT/q*ln(A/A0) |
| Isc | not resolved | Isc(t,T) = Isc_ref*(A(t)/A0)*(1+alpha_Isc*(T-Tref)) |
| Fill factor | constant | FF(t) = FF0*(1 - FF_decay*t), floor at 0.5*FF0 |
| Temperature | none | Full temperature sweep |

## Inputs

| Name | Unit | Default | Range |
|------|------|---------|-------|
| t_years | years | 0.0 | 0–500 |
| T_cell_K | K | 300 | 200–500 |

## Outputs

| Name | Unit | Description |
|------|------|-------------|
| activity_Bq | Bq | Current isotope activity |
| P_beta_absorbed_W | W | Absorbed beta particle power |
| Isc_uA | µA | Short-circuit current |
| Voc_V | V | Open-circuit voltage |
| FF | - | Fill factor |
| P_out_W | W | Maximum power output |
| P_out_uW | µW | Power in microwatts |
| eta_junction | - | Cell conversion efficiency |
| fraction_remaining | - | Activity fraction A(t)/A0 |

## Design Point (Ni-63 diamond cell, t=0, T=300K)

- Isc ≈ 0.5 µA, Voc ≈ 2.0 V, FF ≈ 0.70
- P_out ≈ 0.7 µW, eta_junction ≈ 6% of absorbed power
- After 50 years: P_out ≈ 0.5 µW (Ni-63 retains ~71% activity)

## References

- Olsen, L.C. et al. (1993). *Nucl. Instrum. Methods Phys. Res. B*, 73(1), 139.
- Sychov, M. et al. (2008). *Appl. Radiat. Isot.* 66(2), 173.
- Prelas, M. et al. (2014). *Progress in Nuclear Energy*, 75, 117.
- Sun, W. et al. (2018). *Applied Energy*, 225, 390.
