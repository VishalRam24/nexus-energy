# EC109 -- Simple Cycle Gas Turbine -- F2a Brayton Cycle

## Model Card

| Field | Value |
|-------|-------|
| Component | Simple Cycle Gas Turbine |
| EC ID | EC109 |
| Fidelity | F2a -- Brayton Cycle with Temperature-Dependent Properties |
| Version | 1.0.0 |

## Physics

Open Brayton cycle: compressor + combustor + turbine.

**Temperature-dependent specific heat:**
```
cp_air(T) = 1005 + 0.1*(T - 300) [J/(kg.K)]
```

**Compressor:** T2 = T1 * (P2/P1)^((gamma-1)/(gamma*eta_c))

**Combustor:** m_dot * cp * (T3 - T2) = m_fuel * LHV * eta_comb

**Turbine:** T4 = T3 * (1 - eta_t * (1 - (P4/P3)^((gamma-1)/gamma)))

Features: PR sweep, TIT sweep, part-load via TIT modulation, ambient temperature correction.

## Default Unit: 40 MW Industrial Gas Turbine (LM6000 class)
- PR = 30, TIT = 1250 C, m_dot_air = 130 kg/s
- eta_comp = 0.87, eta_turb = 0.89
- Target electrical efficiency: 30-40%

## Inputs

| Parameter | Unit | Default | Range |
|-----------|------|---------|-------|
| TIT_K | K | 1523.15 | [1073, 1873] |
| PR | - | 30 | [5, 40] |
| m_dot_air | kg/s | 130 | [20, 500] |
| T_amb_K | K | 288.15 | [253, 323] |
| load_fraction | - | 1.0 | [0.3, 1.0] |

## Outputs

| Output | Unit | Description |
|--------|------|-------------|
| W_elec_MW | MW | Electrical power output |
| eta_electrical | - | Electrical efficiency |
| eta_thermal | - | Thermal (shaft) efficiency |
| heat_rate_kJ_kWh | kJ/kWh | Heat rate |
| T_exhaust_K | K | Exhaust temperature |
| SFC_kg_kWh | kg/kWh | Specific fuel consumption |

## Limitations
- Steady-state only
- Simple linear cp(T) model (valid 200-1500 K)
- No blade cooling model
- Part-load by TIT modulation only (no IGV)
- Single-shaft configuration assumed

## References
- Saravanamuttoo et al. (2017), Gas Turbine Theory
- Walsh & Fletcher (2004), Gas Turbine Performance
