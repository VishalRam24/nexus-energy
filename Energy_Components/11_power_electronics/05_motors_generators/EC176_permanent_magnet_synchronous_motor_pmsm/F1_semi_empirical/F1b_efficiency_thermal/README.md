# EC176 — PMSM — F1b Efficiency + Thermal

## Overview
Extends F1a loss-separation model with **PM flux demagnetization** (NdFeB temperature coefficient) and **copper resistance temperature dependence**.

## Physics
- PM flux: `Phi_m(T) = Phi_m_ref * (1 + alpha_Br * (T_mag - T_ref))`, alpha_Br = -0.0012/K
- Stator resistance: `R_s(T) = R_s_ref * (1 + alpha_Cu * (T - T_ref))`
- Torque constant: `k_t(T) proportional to Phi_m(T)` -- drops with temperature
- Lower flux => higher current for same torque => more I2R loss => lower efficiency
- Back-EMF: `E = Phi_m(T) * p * omega_mech`
- Irreversible demagnetization risk above 150C for NdFeB magnets

## Inputs
| Parameter | Unit | Default | Range |
|-----------|------|---------|-------|
| torque | Nm | - | [0, 25] |
| speed_rpm | rpm | - | [0, 12000] |
| magnet_temperature | degC | 80 | [20, 180] |
| ambient_temperature | degC | 25 | [-20, 60] |

## Outputs
| Parameter | Unit |
|-----------|------|
| efficiency | - |
| output_power_kw | kW |
| input_power_kw | kW |
| total_losses_kw | kW |
| torque_Nm | Nm |
| back_emf_V | V |
| derating_factor | - |
| demag_risk | boolean |

## Default Parameters
- Rated power: 5 kW, Rated torque: 16 Nm at 3000 rpm
- Phi_m_ref: 0.3 Wb at 25C, R_s_ref: 0.2 ohm at 25C
- 4 pole pairs, NdFeB magnets

## References
- Gieras, J.F. (2010). Permanent Magnet Motor Technology, 3rd ed. CRC Press.
- Sebastian, T. (1995). IEEE Trans. Magnetics, 31(4), 2578-2584.
