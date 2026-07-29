# EC175 -- Induction Motor/Generator -- F2a dq-Frame Dynamic Model

## Model Description

Full dq synchronous reference frame dynamic model with 4 electrical ODEs (stator and rotor dq currents) plus 1 mechanical ODE (rotor speed).

## Physics

**States:** `x = [i_ds, i_qs, i_dr, i_qr, omega_r]`

```
di_ds/dt = (v_ds - Rs*i_ds + omega_s*Ls*i_qs + omega_s*Lm*i_qr) / Ls
di_qs/dt = (v_qs - Rs*i_qs - omega_s*Ls*i_ds - omega_s*Lm*i_dr) / Ls
di_dr/dt = (-Rr*i_dr + (omega_s-omega_r)*Lr*i_qr + (omega_s-omega_r)*Lm*i_qs) / Lr
di_qr/dt = (-Rr*i_qr - (omega_s-omega_r)*Lr*i_dr - (omega_s-omega_r)*Lm*i_ds) / Lr
T_e = 1.5 * P * Lm * (i_qs*i_dr - i_ds*i_qr)
d(omega_r)/dt = (T_e - T_load - B*omega_r) / J
```

## Parameters

| Parameter | Value | Unit |
|-----------|-------|------|
| Rs | 0.5 | Ohm |
| Rr | 0.4 | Ohm |
| Ls | 0.08 | H |
| Lr | 0.08 | H |
| Lm | 0.075 | H |
| P | 2 | pole pairs |
| J | 0.1 | kg.m2 |
| B | 0.001 | N.m.s/rad |

## Inputs / Outputs

**Inputs:** v_supply_rms, frequency_hz, T_load_Nm, dt, duration_s
**Outputs:** t, speed_rpm, torque, current, power, slip

## Reference

Boldea, I. & Nasar, S.A. (2010). *The Induction Machine Handbook*, 2nd ed. CRC Press.
