# EC157 — Buck Converter (Step-Down) — F1a Ideal Gain + Losses

## Overview
Semi-empirical model for a synchronous buck (step-down) DC-DC converter. Covers ideal
voltage gain, conduction losses (MOSFET + inductor + diode), and switching transition losses.

## Model Equations

**Ideal voltage conversion:**
```
D = V_out / V_in
V_out = D * V_in
```

**Conduction losses:**
```
P_cond = I_out^2 * (Rds_on*D + R_L + Vd*(1-D)/V_out)
```

**Switching losses:**
```
P_sw = 0.5 * V_in * I_out * (t_on + t_off) * f_sw
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
| V_in | 48 | V | Nominal input voltage |
| V_out | 12 | V | Nominal output voltage |
| f_sw | 100 | kHz | Switching frequency |
| Rds_on | 10 | mΩ | MOSFET on-resistance |
| R_L | 50 | mΩ | Inductor DCR |
| V_diode | 0.5 | V | Freewheeling diode Vf |
| t_on = t_off | 30 | ns | Transition times |
| I_rated | 10 | A | Rated output current |

## Inputs
| Name | Unit | Range | Description |
|------|------|-------|-------------|
| v_in | V | [20, 100] | Input bus voltage |
| v_out_target | V | [1, 50] | Target output voltage |
| i_load | A | [0.1, 20] | Output load current |

## Outputs
| Name | Unit | Description |
|------|------|-------------|
| duty_cycle | — | D = V_out / V_in |
| v_out | V | Actual output voltage (ideal = target) |
| efficiency | — | Overall power efficiency |
| p_loss_w | W | Total losses |
| p_conduction_w | W | Conduction losses |
| p_switching_w | W | Switching losses |

## Validation Results (V_in=48V, V_out=12V)
| I_load (A) | D | eta (%) | P_cond (W) | P_sw (W) |
|-----------|---|---------|------------|---------|
| 0.5 | 0.25 | 89.7 | 0.013 | 0.072 |
| 1.0 | 0.25 | 94.4 | 0.051 | 0.144 |
| 2.0 | 0.25 | 96.8 | 0.206 | 0.288 |
| 5.0 | 0.25 | 97.7 | 1.288 | 0.720 |
| 10.0 | 0.25 | 97.1 | 5.150 | 1.440 |

## Sources
1. Erickson, R.W. & Maksimovic, D. (2020). *Fundamentals of Power Electronics*, 3rd ed. Springer.

## Limitations
- Ideal voltage gain (no line/load regulation)
- No ripple or inductor current model
- No thermal model or derating
- Synchronous rectification approximated as diode loss term
