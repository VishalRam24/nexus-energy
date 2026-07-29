# EC025 Lithium-Sulfur Battery — F1b SOC-Thermal Model

## Overview

Semi-empirical voltage model for Lithium-Sulfur (Li-S) cells with temperature dependence.

**Fidelity:** F1b — SOC + Thermal  
**Chemistry:** Lithium metal anode, sulfur cathode (carbon-sulfur composite)  
**Reference cell:** Generic Li-S pouch, 3.0 Ah

## Critical Physics — Positive dOCV/dT (UNIQUE to Li-S)

Li-S is unique among battery chemistries: **dOCV/dT = +0.35 mV/K (positive)**.

This is because the overall discharge reaction Li + 0.5S → 0.5Li2S has a positive
entropy change (ΔS > 0). By the thermodynamic identity:
    dOCV/dT = ΔS / (n·F)
a positive ΔS gives positive dOCV/dT.

Consequence on heat generation:
    Q = I²R(T) + I·T·dOCV/dT
- During discharge (I > 0): both terms positive → net exothermic
- During charge (I < 0): reversible term becomes negative (endothermic cooling)

This contrasts with Li-ion cathodes (NMC, LFP, LMO) where dOCV/dT < 0,
making discharge partially endothermic at low currents.

Reference: Kumaresan et al. (2008), J. Electrochem. Soc. 155(6), A576.

## Physics Added Over F1a

| Feature | Equation | Notes |
|---------|----------|-------|
| Arrhenius resistance | R(T) = R_ref * exp(Ea/R * (1/T - 1/T_ref)) | Ea = 30 kJ/mol |
| dOCV/dT | +0.35 mV/K | POSITIVE — unique to Li-S |
| Heat generation | Q = I²R(T) + I·T·dOCV/dT | Net exothermic during discharge |
| Capacity correction | C(T) = C_ref * (1 + alpha_c*(T-T_ref)) | alpha_c = 0.008 /K (strong T dependence) |

## Data Sources

- Wild et al. (2015). Energy Environ. Sci. 8, 3477.
- Mikhaylik & Akridge (2004). J. Electrochem. Soc. 151, A1969.
- Kumaresan et al. (2008). J. Electrochem. Soc. 155(6), A576.
- Cuisinier et al. (2014). J. Phys. Chem. Lett. 5, 3227.

## Limitations

- Does not capture polysulfide shuttle (main capacity fade mechanism for Li-S)
- Simplified single-curve OCV (actual Li-S has a distinct two-plateau shape)
- Valid for -15 to 60 °C, up to ~2C rate
- Lithium metal anode not explicitly modeled (no dendrite risk)
