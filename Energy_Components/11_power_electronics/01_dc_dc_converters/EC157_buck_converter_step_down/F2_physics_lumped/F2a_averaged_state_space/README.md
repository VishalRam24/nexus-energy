# EC157 -- Buck Converter (Step-Down) -- F2a Averaged State-Space Model

## Model Description

Averaged continuous-time state-space ODE model for a buck (step-down) DC-DC converter operating in continuous conduction mode (CCM). The switching dynamics are averaged over one switching cycle, yielding a smooth ODE system suitable for control design and transient analysis.

## Physics

**States:** `x = [i_L, v_C]` (inductor current, capacitor voltage)

**State equations:**
```
di_L/dt = (D * V_in - v_C - i_L * R_L) / L
dv_C/dt = (i_L - v_C / R_load) / C
```

where `D` is the duty cycle, `V_in` is the input voltage, `R_L` is the inductor ESR, `L` is the inductance, `C` is the output capacitance, and `R_load` is the load resistance.

## Parameters

| Parameter | Value | Unit | Description |
|-----------|-------|------|-------------|
| L | 100 | uH | Inductance |
| C | 100 | uF | Output capacitance |
| R_L | 0.05 | Ohm | Inductor ESR |
| f_sw | 100 | kHz | Switching frequency |

## Inputs

| Name | Unit | Range | Description |
|------|------|-------|-------------|
| v_in | V | 10--100 | Input DC voltage |
| duty_cycle | -- | 0.05--0.95 | Duty cycle |
| R_load | Ohm | 0.5--100 | Load resistance |
| dt | s | 1e-7 -- 1e-3 | Simulation time step |
| duration_s | s | 1e-4 -- 1.0 | Simulation duration |

## Outputs

| Name | Unit | Description |
|------|------|-------------|
| t | s | Time array |
| v_out | V | Output voltage (= v_C) |
| i_L | A | Inductor current |
| i_out | A | Output current (= v_C / R_load) |
| power | W | Output power |

## Limitations

- Averaged model: no switching ripple captured (use F2b for switching-level)
- Assumes CCM operation (inductor current does not reach zero)
- No thermal effects (use F2c for thermal coupling)

## Reference

Erickson, R.W. & Maksimovic, D. (2020). *Fundamentals of Power Electronics*, 3rd ed. Springer. Chapter 7.
