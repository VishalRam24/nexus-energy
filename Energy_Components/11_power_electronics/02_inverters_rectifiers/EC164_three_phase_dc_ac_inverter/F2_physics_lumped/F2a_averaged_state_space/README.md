# EC164 -- Three-Phase DC-AC Inverter -- F2a dq-Frame Averaged Model

## Model Description

dq synchronous reference frame averaged model for a grid-tied three-phase voltage source inverter with PI current control.

## Physics

**States:** `x = [i_d, i_q]` (+ PI integrator states)

```
di_d/dt = (v_d - R*i_d + omega_e*L*i_q - e_d) / L
di_q/dt = (v_q - R*i_q - omega_e*L*i_d - e_q) / L
```

**Power:** `P = 1.5*(e_d*i_d + e_q*i_q)`, `Q = 1.5*(e_q*i_d - e_d*i_q)`

## Parameters

| Parameter | Value | Unit |
|-----------|-------|------|
| L | 2 | mH |
| R | 0.1 | Ohm |
| V_dc | 800 | V |
| f_grid | 50 | Hz |
| V_grid_rms | 400 | V (line-to-line) |

## Inputs / Outputs

**Inputs:** P_ref_kw, Q_ref_kvar, dt, duration_s
**Outputs:** t, i_d, i_q, P, Q, v_dc

## Reference

Teodorescu, R., Liserre, M. & Rodriguez, P. (2011). *Grid Converters for Photovoltaic and Wind Power Systems*. Wiley.
