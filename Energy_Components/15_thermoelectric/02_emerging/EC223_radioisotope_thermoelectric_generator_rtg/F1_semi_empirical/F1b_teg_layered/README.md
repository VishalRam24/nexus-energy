# EC223 — RTG — F1b Multi-Layer SiGe TEG Model

**Fidelity:** F1b — Semi-Empirical  
**Sub-branch:** Temperature-dependent SiGe properties, thermal resistance network, self-consistent junction temperatures

## Model Description

Extends F1a (parametric decay + degradation factor) with:

| Feature | F1a | F1b |
|---------|-----|-----|
| Material properties | constant ZT=0.5 | alpha(T), k(T), sigma(T) for SiGe |
| Junction temperatures | T_hot from radiative scaling only | thermal resistance network (hot shoe + radiator) |
| Efficiency | Angist at nominal T_hot/T_cold | Angist with ZT integrated over actual junction dT |
| Electrical outputs | P_electric only | V_oc, I_mp, R_int, P_max_circuit |
| Self-consistency | none | iterative solve for T_cj(P_radiated) convergence |

## Physics

**SiGe material model** (T_ref = 1073 K):
- alpha(T) = 210e-6 * (1 + a1*(T-T_ref))  [V/K]
- k(T) = 3.5 * (1 + b1*(T-T_ref))  [W/(m*K)]
- sigma(T) = 5e4 * (1 + c1*(T-T_ref))  [S/m]
- ZT(T) = alpha(T)^2 * sigma(T) * T / k(T)  ≈ 0.6 at 1073K

**Thermal resistance network**:
- T_hj = T_source - P_thermal * R_hot(t)  (contact degradation increases R_hot)
- T_cj = T_cold_sink + P_radiated * R_cold
- Self-consistent with 12-iteration converge to <0.05 K

## Inputs

| Name | Unit | Default | Range |
|------|------|---------|-------|
| t_years | years | 0.0 | 0–200 |

## Outputs

| Name | Unit | Description |
|------|------|-------------|
| P_thermal_W | W | Pu-238 decay thermal power |
| T_hj_K | K | Hot junction temperature |
| T_cj_K | K | Cold junction temperature |
| ZT_avg | - | ZT averaged over junction dT |
| eta_teg | - | TEG conversion efficiency |
| eta_carnot | - | Carnot efficiency at junction T |
| P_electric_W | W | Electrical output power |
| P_max_circuit_W | W | Matched-load circuit power |
| V_oc_V | V | Open-circuit voltage |
| I_mp_A | A | Matched-load current |
| R_int_ohm | Ω | Module internal resistance |

## Design Point (GPHS-RTG analog, t=0)

- P_thermal = 4500 W, P_electric ≈ 207 W, eta_teg ≈ 4.6%
- ZT_avg ≈ 0.60 (SiGe n-type at ~1073K reference)
- T_hj ≈ 1183 K, T_cj ≈ 788 K
- After 50 years: P_electric ≈ 140 W (~68% of BOL)

## References

- Bennett, G.L. (2006). Space nuclear power. *Acta Astronautica*.
- El-Genk, M.S. & Saber, H.H. (2005). *Energy Convers. Mgmt.* 46(7-8), 1083.
- Fleurial, J-P. et al. (1997). SiGe unicouples. *Proc. 16th IECEC*.
- Rowe, D.M. (ed.) (2006). *Thermoelectrics Handbook*. CRC Press.
