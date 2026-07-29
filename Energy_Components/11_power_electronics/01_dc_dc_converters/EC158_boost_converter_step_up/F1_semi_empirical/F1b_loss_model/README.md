# EC158 -- Boost Converter (Step-Up) -- F1b Detailed Semiconductor Loss Model

## Overview
Extends F1a by decomposing converter losses into four distinct physical mechanisms:
MOSFET conduction, diode conduction, switching transitions, and inductor DCR.
Boost topology has higher input current (I_in = I_out * V_out / V_in), so inductor
and MOSFET losses are amplified relative to buck.

## Model Equations

**MOSFET conduction loss:**
```
P_cond_mosfet = I_in^2 * D * R_ds_on
```

**Diode conduction loss:**
```
P_cond_diode = I_out * V_f
```

**Switching loss (MOSFET switches at V_out):**
```
P_sw = 0.5 * V_out * I_in * (t_on + t_off) * f_sw
```

**Inductor DCR loss:**
```
P_L = I_in^2 * R_L
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

## Sources
1. Erickson, R.W. & Maksimovic, D. (2020). *Fundamentals of Power Electronics*, 3rd ed. Springer.

## Limitations
- Ideal voltage gain (no parasitic effects on gain)
- No ripple current model
- No thermal derating
- D clipped to 0.95 max (extreme duty cycles not modeled)
