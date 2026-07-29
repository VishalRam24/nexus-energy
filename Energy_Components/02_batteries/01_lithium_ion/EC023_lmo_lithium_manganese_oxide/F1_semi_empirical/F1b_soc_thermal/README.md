# EC023 LMO Battery — F1b SOC-Thermal Model

## Overview

Semi-empirical voltage model for Lithium Manganese Oxide (LMO / spinel LiMn2O4) cells
that adds temperature dependence on top of the F1a SOC-only baseline.

**Fidelity:** F1b — SOC + Thermal  
**Chemistry:** LMO spinel cathode, graphite anode  
**Reference cell:** Generic LMO 18650, 3.0 Ah

## Physics Added Over F1a

| Feature | Equation | Notes |
|---------|----------|-------|
| Arrhenius resistance | R(T) = R_ref * exp(Ea/R * (1/T - 1/T_ref)) | Ea = 23 kJ/mol |
| OCV temperature shift | OCV_eff = OCV(SOC) + dOCV/dT * (T - T_ref) | dOCV/dT = -0.18 mV/K |
| Heat generation | Q = I²R(T) + I·T·dOCV/dT | Joule + entropic |
| Capacity correction | C(T) = C_ref * (1 + alpha_c * (T - T_ref)) | alpha_c = 0.004 /K |

## Chemistry Note — dOCV/dT

LMO spinel has a mild negative entropic coefficient of approximately -0.18 mV/K,
significantly less negative than NMC811 (-0.40 mV/K) or NCA (-0.35 mV/K).
This is because the Mn spinel lattice accommodates lithium intercalation with
less structural entropy change than layered oxide cathodes.
Reference: Thomas & Newman (2003), J. Electrochem. Soc. 150, A176.

## Inputs / Outputs

| Parameter | Unit | Range | Notes |
|-----------|------|--------|-------|
| soc | — | 0–1 | State of charge |
| current | A | -15 to 15 | Positive = discharge |
| temperature | K | 253–333 | -20 to 60 °C |

| Output | Unit | Description |
|--------|------|-------------|
| terminal_voltage | V | V = OCV - I·R(T), clipped to [3.0, 4.2] V |
| power | W | Electrical power (positive = discharge) |
| heat_generation | W | Total heat (Joule + entropic) |
| effective_capacity | Ah | Temperature-corrected capacity |
| internal_resistance | Ohm | Arrhenius R(T) |
| ocv | V | Open-circuit voltage at SOC |
| dsoc_dt | 1/s | SOC rate of change |

## Data Sources

- Liaw et al. (2003). J. Power Sources, 119-121, 874-882. (OCV, R parameters)
- Jalkanen et al. (2015). Applied Energy 154, 160-172. (thermal capacity coefficient)
- Thomas & Newman (2003). J. Electrochem. Soc. 150, A176. (entropic coefficient)
- Ecker et al. (2015). J. Electrochem. Soc., 162(9), A1836. (activation energy)

## Limitations

- Single flat voltage plateau; does not capture LMO double-plateau structure
- No capacity fade from Mn dissolution (see F1c/F1d)
- Lumped thermal model; no spatial temperature gradient
- Valid for -20 to 60 °C, 0.1C to ~5C rates
