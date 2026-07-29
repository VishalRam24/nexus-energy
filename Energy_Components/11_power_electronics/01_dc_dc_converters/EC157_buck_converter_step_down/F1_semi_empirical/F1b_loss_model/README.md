# EC157 -- Buck Converter (Step-Down) -- F1b Detailed Semiconductor Loss Model

## Overview
Extends F1a by decomposing converter losses into four distinct physical mechanisms:
MOSFET conduction, diode conduction, switching transitions, and inductor DCR.

## Model Equations

**MOSFET conduction loss:**
```
P_cond_mosfet = I_out^2 * D * R_ds_on
```

**Diode conduction loss:**
```
P_cond_diode = I_out * (1 - D) * V_f
```

**Switching loss:**
```
P_sw = 0.5 * V_in * I_out * (t_on + t_off) * f_sw
```

**Inductor DCR loss:**
```
P_L = I_out^2 * R_L
```

**Efficiency:**
```
P_out = V_out * I_out
P_loss = P_cond_mosfet + P_cond_diode + P_sw + P_L
eta = P_out / (P_out + P_loss)
```

## Default Parameters
| Parameter | Value | Unit | Description |
|-----------|-------|------|-------------|
| R_ds_on | 10 | mohm | MOSFET on-resistance |
| V_f | 0.5 | V | Diode forward voltage |
| t_on | 25 | ns | MOSFET turn-on time |
| t_off | 35 | ns | MOSFET turn-off time |
| f_sw | 100 | kHz | Switching frequency |
| R_L | 20 | mohm | Inductor DCR |

## Inputs / Outputs
Same inputs as F1a (v_in, v_out_target, i_load). Outputs add per-component loss breakdown.

## Sources
1. Erickson, R.W. & Maksimovic, D. (2020). *Fundamentals of Power Electronics*, 3rd ed. Springer.

## Limitations
- Ideal voltage gain (no line/load regulation)
- No ripple current model (assumes continuous conduction mode)
- No thermal derating of R_ds_on or V_f
- No gate drive losses
