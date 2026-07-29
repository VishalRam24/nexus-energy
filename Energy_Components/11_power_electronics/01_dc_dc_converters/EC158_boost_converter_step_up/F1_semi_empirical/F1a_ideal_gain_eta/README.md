# EC158 — Boost Converter (Step-Up) — F1a Ideal Gain + Losses

## Overview
Semi-empirical model for a boost (step-up) DC-DC converter. Models the ideal voltage
conversion gain, input current amplification, and conduction/switching losses adapted
for boost topology.

## Model Equations

**Ideal voltage conversion:**
```
D = 1 - V_in / V_out
V_out = V_in / (1 - D)
```

**Input current (from power conservation):**
```
I_in = I_out * V_out / V_in
```

**Conduction losses:**
```
P_cond = I_in^2 * (Rds_on*D + R_L) + I_out * Vd
```

**Switching losses:**
```
P_sw = 0.5 * V_out * I_in * (t_on + t_off) * f_sw
```

**Efficiency:**
```
P_out = V_out * I_out
P_in  = P_out + P_cond + P_sw
eta   = P_out / P_in
```

## Default Parameters
| Parameter | Value | Unit | Description |
|-----------|-------|------|-------------|
| V_in | 12 | V | Nominal input voltage |
| V_out | 48 | V | Nominal output voltage |
| f_sw | 100 | kHz | Switching frequency |
| Rds_on | 15 | mΩ | MOSFET on-resistance |
| R_L | 80 | mΩ | Inductor DCR |
| V_diode | 0.5 | V | Output diode Vf |
| t_on = t_off | 30 | ns | Transition times |
| I_rated (out) | 5 | A | Rated output current |

## Inputs
| Name | Unit | Range | Description |
|------|------|-------|-------------|
| v_in | V | [5, 40] | Input voltage |
| v_out_target | V | [10, 200] | Target output voltage |
| i_load | A | [0.1, 15] | Output (load-side) current |

## Outputs
| Name | Unit | Description |
|------|------|-------------|
| duty_cycle | — | D = 1 - V_in/V_out |
| v_out | V | Actual output voltage |
| efficiency | — | Overall power efficiency |
| p_loss_w | W | Total losses |
| i_input | A | Input (inductor) current |

## Validation Results (V_in=12V, V_out=48V)
| I_out (A) | D | eta (%) | I_in (A) | P_loss (W) |
|-----------|---|---------|----------|-----------|
| 0.5 | 0.75 | 93.0 | 2.0 | 1.79 |
| 1.0 | 0.75 | 93.5 | 4.0 | 3.36 |
| 2.0 | 0.75 | 93.3 | 8.0 | 6.89 |
| 5.0 | 0.75 | 91.5 | 20.0 | 22.3 |

## Key Characteristics
- **Current amplification**: I_in ≈ M * I_out where M = V_out/V_in
- **Duty cycle at 4:1 boost**: D = 0.75
- **Switching loss dominates at high V_out** (scales with V_out * I_in)
- **Conduction loss dominates at high I** (scales with I_in² for MOSFET/inductor)

## Sources
1. Erickson, R.W. & Maksimovic, D. (2020). *Fundamentals of Power Electronics*, 3rd ed. Springer.

## Limitations
- Ideal voltage gain (no discontinuous conduction mode, no right-half-plane zero)
- No ripple or inductor current model
- No thermal model
- Duty cycle clamped at 0.95 (avoid D→1 singularity)
