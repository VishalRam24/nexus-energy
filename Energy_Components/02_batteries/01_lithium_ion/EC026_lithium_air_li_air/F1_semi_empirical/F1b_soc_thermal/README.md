# EC026 Lithium-Air Battery — F1b SOC-Thermal Model

## Overview

Semi-empirical voltage model for aprotic Lithium-Air (Li-O2) cells with temperature dependence.

**Fidelity:** F1b — SOC + Thermal  
**Chemistry:** Lithium metal anode, carbon air cathode (aprotic electrolyte)  
**Reference cell:** Aprotic Li-O2 pouch, 1.0 Ah

## Critical Physics — Largest Negative dOCV/dT

Li-air has dOCV/dT ≈ **-0.50 mV/K**, the most negative of common battery chemistries.

This reflects the large entropy decrease during Li2O2 formation:
    2Li + O2 → Li2O2    (ΔS strongly negative due to gas-phase O2 ordering)
    dOCV/dT = ΔS / (n·F) < 0

Consequence: during discharge (I > 0), both Joule and entropic heating are positive.
Li-air cells generate significantly more heat per unit energy than Li-ion cells.

| Chemistry | dOCV/dT (mV/K) |
|-----------|----------------|
| Li-air    | -0.50          |
| NMC       | -0.40          |
| NCA       | -0.35          |
| LFP       | -0.25          |
| LMO       | -0.18          |
| Li-S      | +0.35          |

## Physics Added Over F1a

| Feature | Equation | Notes |
|---------|----------|-------|
| Arrhenius resistance | R(T) = R_ref * exp(Ea/R * (1/T - 1/T_ref)) | Ea = 35 kJ/mol (highest due to ORR kinetics) |
| dOCV/dT | -0.50 mV/K | Most negative, adds to discharge heating |
| Heat generation | Q = I²R(T) + I·T·dOCV/dT | Both terms positive during discharge |
| Capacity correction | C(T) = C_ref * (1 + alpha_c*(T-T_ref)) | alpha_c = 0.006 /K |

## Data Sources

- Abraham & Jiang (1996). J. Electrochem. Soc. 143, 1.
- Laoire et al. (2010). J. Electrochem. Soc. 157(7), A821.
- Viswanathan et al. (2011). J. Chem. Phys. 135, 214704.
- Lu et al. (2013). Nat. Chem. 5, 527.

## Limitations

- Does not capture Li2O2 pore clogging (capacity fade) — use F1c/F1d
- No oxygen transport modeling
- Theoretical capacity assumes complete Li2O2 formation
- Valid for -15 to 60 °C, very low C-rates (≤2C)
