# EC176 — PMSM — F1a Efficiency Map Model

## Overview
Loss-separation efficiency map for a Permanent Magnet Synchronous Motor (PMSM).
P_copper = (T/k_t)^2 * R_s, P_iron = k_e * omega^1.5, P_mech = k_f * omega.
eta = P_out / (P_out + P_loss).

## Inputs
| Name | Unit | Range | Description |
|------|------|-------|-------------|
| torque | Nm | [0, 200] | Output shaft torque |
| speed_rpm | rpm | [0, 12000] | Rotor speed |

## Outputs
| Name | Unit | Description |
|------|------|-------------|
| efficiency | - | Motor efficiency |
| output_power_kw | kW | Mechanical output power |
| input_power_kw | kW | Electrical input power |
| total_losses_kw | kW | Sum of all losses |

## Parameters (Calibrated)
- P_rated: 50 kW
- T_rated: 160 Nm
- omega_base: 3000 rpm
- omega_max: 12000 rpm
- eta_peak: 0.96 (at T=160Nm, 3000rpm — verified)
- k_t: 0.5093 Nm/A (torque constant)
- R_s: 0.008 ohm (stator resistance)
- k_e: 0.005559 W/rpm^1.5 (iron loss, calibrated)
- k_f: 0.130484 W/rpm (mechanical loss, calibrated)

## Loss Model
- Copper: dominates at high torque (I^2*R)
- Iron: proportional to omega^1.5, dominates at high speed low torque
- Mechanical: proportional to omega (friction + windage)

## Sources
1. Gieras, J.F. (2010). Permanent Magnet Motor Technology, 3rd ed. CRC Press. Chapter 3.
2. Morimoto, S. et al. (1994). IEEE Trans. Ind. Appl., 30(4), 920-926.

## Limitations
- No field-weakening model (constant-flux assumed throughout speed range)
- No thermal derating (constant parameters regardless of winding temperature)
- No saturation effects or cross-coupling
- Parameters calibrated for a generic 50kW motor; scale k_e/k_f for other sizes
