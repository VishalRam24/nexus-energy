# EC164 -- Three-Phase DC-AC Inverter -- F1b Detailed IGBT/Diode Loss Model

## Overview
Extends F1a (part-load efficiency curve) with physics-based IGBT and diode loss calculations
for a 6-device (3 half-bridge) voltage-source inverter. Uses sinusoidal PWM analytical
current distribution formulas from Semikron application notes.

## Model Equations

**IGBT conduction (per device):**
```
P_cond_igbt = V_ce0 * I_avg + r_ce * I_rms^2
I_avg  = I_peak / (2*pi) * (1 + m*pi*cos(phi)/4)
I_rms^2 = I_peak^2 * (1/8 + m*cos(phi)/(3*pi))
```

**IGBT switching (per device):**
```
P_sw_igbt = (E_on + E_off) * f_sw * (V_dc/V_ref) * (I_peak/(pi*I_ref))
```

**Diode conduction (per device):**
```
P_cond_diode = V_f * I_avg_d + r_d * I_rms_d^2
```

**Diode reverse recovery (per device):**
```
P_rr = E_rr * f_sw * (V_dc/V_ref) * (I_peak/(pi*I_ref))
```

**Total: multiply by 6 for full bridge.**

## Default Parameters
| Parameter | Value | Unit | Description |
|-----------|-------|------|-------------|
| V_ce0 | 1.5 | V | IGBT threshold voltage |
| r_ce | 2 | mohm | IGBT slope resistance |
| E_on | 15 | mJ | Turn-on energy |
| E_off | 10 | mJ | Turn-off energy |
| V_f | 1.2 | V | Diode forward voltage |
| r_d | 1 | mohm | Diode slope resistance |
| E_rr | 8 | mJ | Reverse recovery energy |
| f_sw | 10 | kHz | Switching frequency |
| V_ref | 600 | V | Reference voltage for loss data |
| I_ref | 100 | A | Reference current for loss data |
| V_dc | 800 | V | DC bus voltage |

## Sources
1. Semikron Application Manual (2015), Power Semiconductors.
2. Mohan, Undeland & Robbins (2003), *Power Electronics*, 3rd ed. Wiley.

## Limitations
- No thermal feedback on V_ce0, r_ce (junction temperature assumed constant)
- Dead-time effects not modeled
- No output filter losses
- Sinusoidal PWM analytical formulas (not SVPWM exact distribution)
