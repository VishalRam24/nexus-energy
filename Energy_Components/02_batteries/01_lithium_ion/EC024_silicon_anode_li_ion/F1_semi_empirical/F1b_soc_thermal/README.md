# EC024 Silicon-Anode Li-ion Battery — F1b SOC-Thermal Model

## Overview

Semi-empirical voltage model for silicon-blend anode Li-ion cells (Si/NMC, ~10% Si by weight).

**Fidelity:** F1b — SOC + Thermal  
**Chemistry:** NMC cathode, Si-graphite blend anode  
**Reference cell:** Si/NMC pouch, 3.5 Ah

## Physics Added Over F1a

| Feature | Equation | Notes |
|---------|----------|-------|
| Arrhenius resistance | R(T) = R_ref * exp(Ea/R * (1/T - 1/T_ref)) | Ea = 26 kJ/mol (higher than NMC due to Si SEI) |
| OCV temperature shift | dOCV/dT = -0.15 mV/K | Less negative than pure NMC/graphite |
| Heat generation | Q = I²R(T) + I·T·dOCV/dT | Joule + entropic |
| Capacity correction | C(T) = C_ref * (1 + alpha_c*(T-T_ref)) | alpha_c = 0.005 /K |

## Chemistry Note — dOCV/dT

Si-blend anodes shift dOCV/dT slightly less negative than pure graphite/NMC (-0.40 mV/K)
because the silicon intercalation plateau (~0.4 V vs Li/Li+) has a positive entropic
contribution that partially offsets the graphite contribution.
Average over SOC range: approximately -0.15 mV/K.
Reference: Geng et al. (2020), J. Electrochem. Soc. 167, 090504.

## Chemistry Note — E_a

Higher activation energy (26 kJ/mol vs ~25 for NMC) reflects the additional
impedance from SEI layer cracking at the silicon particles during expansion cycles.

## Data Sources

- Zheng et al. (2014). J. Electrochem. Soc. 161(11), A2066.
- McDowell et al. (2013). Adv. Mater. 25, 4966.
- Geng et al. (2020). J. Electrochem. Soc. 167, 090504.
- Thomas & Newman (2003). J. Electrochem. Soc. 150, A176.

## Limitations

- Does not capture silicon expansion-induced resistance growth (use F1c/F1d for aging)
- Simplified OCV curve does not resolve Si plateau explicitly
- Valid for 0.1C to ~5C, -20 to 60 °C
