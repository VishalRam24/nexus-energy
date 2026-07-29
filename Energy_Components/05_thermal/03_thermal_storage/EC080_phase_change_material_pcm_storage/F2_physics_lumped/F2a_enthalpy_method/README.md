# EC080 -- Phase Change Material (PCM) Storage -- F2a Enthalpy Method

## Model Card

| Field | Value |
|-------|-------|
| Component | Phase Change Material (PCM) Storage |
| EC ID | EC080 |
| Fidelity | F2a -- Enthalpy Method (Multi-Node ODE) |
| Version | 1.0.0 |

## Physics

Enthalpy-based ODE with phase change for a 10-node lumped model. Each node tracks specific enthalpy H_i:

```
m_i * dH_i/dt = Q_htf_i + Q_cond_{i-1,i} + Q_cond_{i,i+1} - Q_loss_i
```

Temperature recovery is piecewise:
- **Solid** (H < 0): T = T_melt + H / cp_s
- **Melting** (0 <= H <= L_f): T = T_melt
- **Liquid** (H > L_f): T = T_melt + (H - L_f) / cp_l

HTF coupling uses effectiveness-NTU for co-flow arrangement.

## Default PCM: Paraffin RT60
- T_melt = 60 C, L_f = 180 kJ/kg
- cp_solid = 2.0 kJ/(kg.K), cp_liquid = 2.2 kJ/(kg.K)
- Total mass = 100 kg (10 nodes x 10 kg)

## Inputs

| Parameter | Unit | Default | Range |
|-----------|------|---------|-------|
| T_htf_K | K | 353.15 | [283, 373] |
| m_dot_htf | kg/s | 0.5 | [0, 5] |
| T_init_K | K | 293.15 | [273, 373] |
| dt | s | 10.0 | [0.1, 60] |
| duration_s | s | 3600 | [1, 86400] |

## Outputs

| Output | Unit | Description |
|--------|------|-------------|
| T_mean | K | Mean PCM temperature |
| T_nodes | K | Per-node temperatures (10 x Nt) |
| lf_mean | - | Mean liquid fraction |
| E_stored_J | J | Total energy stored |
| Q_rate_W | W | Charging/discharging power |
| T_htf_out | K | HTF outlet temperature |

## Limitations
- 0D per node (no intra-node gradients)
- Constant material properties (no T-dependence of k, cp)
- Co-flow HTF arrangement only
- No natural convection enhancement in liquid phase

## References
- Voller & Cross (1990), Int. J. Heat Mass Transfer
- Zalba et al. (2003), Applied Thermal Energy, 23(3), 251-283
