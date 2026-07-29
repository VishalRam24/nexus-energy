# EC158 -- Boost Converter (Step-Up) -- F2a Averaged State-Space Model

## Model Description

Averaged continuous-time state-space ODE model for a boost (step-up) DC-DC converter in CCM.

## Physics

**States:** `x = [i_L, v_C]`

```
di_L/dt = (V_in - (1-D)*v_C - i_L*R_L) / L
dv_C/dt = ((1-D)*i_L - v_C/R_load) / C
```

## Parameters

| Parameter | Value | Unit |
|-----------|-------|------|
| L | 200 | uH |
| C | 220 | uF |
| R_L | 0.1 | Ohm |
| f_sw | 100 | kHz |

## Inputs / Outputs

**Inputs:** v_in, duty_cycle, R_load, dt, duration_s
**Outputs:** t, v_out, i_L, i_out, power (time-series arrays)

## Reference

Erickson, R.W. & Maksimovic, D. (2020). *Fundamentals of Power Electronics*, 3rd ed. Springer.
