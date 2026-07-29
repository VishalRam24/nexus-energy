# EC175 — Induction Motor/Generator — F1a Efficiency Map

## Overview
Semi-empirical efficiency map for a 3-phase induction motor based on the IEC 60034-30-1
efficiency classification. Models efficiency as a function of part-load ratio (PLR) using
a polynomial loss decomposition (constant + linear + quadratic terms).

## Model Equations

**Efficiency vs PLR:**
```
loss_norm(PLR) = a0 + a1*PLR + a2*PLR^2
eta(PLR) = eta_rated * loss_norm(1.0) / loss_norm(PLR)
```

Where:
- `a0` = 0.02 — constant losses (iron core, friction, windage)
- `a1` = 0.05 — load-proportional losses (stray load)
- `a2` = 0.93 — copper losses (I²R, proportional to PLR²)

**Power balance:**
```
P_out = PLR * P_rated
P_in  = P_out / eta(PLR)
P_loss = P_in - P_out
```

**Slip (linear approximation):**
```
s(PLR) = s_rated * PLR
```

## Default Parameters
| Parameter | Value | Unit |
|-----------|-------|------|
| Rated power | 15 | kW |
| eta_rated (IE3) | 0.917 | — |
| Power factor | 0.86 | — |
| Poles | 4 | — |
| Frequency | 50 | Hz |
| Sync speed | 1500 | rpm |
| Rated speed | 1460 | rpm |
| Rated slip | 0.0267 | — |

## Inputs
| Name | Unit | Range | Description |
|------|------|-------|-------------|
| load_fraction | — | [0.05, 1.2] | Part-load ratio (PLR) |
| speed_rpm | rpm | [0, 1500] | Rotor speed (optional) |

## Outputs
| Name | Unit | Description |
|------|------|-------------|
| efficiency | — | Overall motor efficiency |
| input_power_kw | kW | Electrical input power |
| output_power_kw | kW | Mechanical shaft output |
| losses_kw | kW | Total electrical losses |
| slip | — | Rotor slip (dimensionless) |

## Validation Results
| PLR | eta (%) | P_in (kW) | P_out (kW) | Losses (kW) | Slip (%) |
|-----|---------|-----------|------------|-------------|----------|
| 0.25 | 88.1 | 4.25 | 3.75 | 0.50 | 0.67 |
| 0.50 | 91.5 | 8.20 | 7.50 | 0.70 | 1.34 |
| 0.75 | 92.0 | 12.22 | 11.25 | 0.97 | 2.00 |
| 1.00 | 91.7 | 16.36 | 15.00 | 1.36 | 2.67 |
| 1.20 | 90.4 | 19.91 | 18.00 | 1.91 | 3.20 |

## Sources
1. IEC 60034-30-1:2014. Rotating electrical machines — Efficiency classes (IE code).
2. Boldea, I. & Nasar, S.A. (2010). *The Induction Machine Handbook*. CRC Press.

## Limitations
- No thermal derating model
- No frequency-dependent speed control (fixed 50 Hz supply)
- Slip model is linear approximation (valid for low-slip operation only)
- Single efficiency curve per rated point (no parameter sweep over frame sizes)
