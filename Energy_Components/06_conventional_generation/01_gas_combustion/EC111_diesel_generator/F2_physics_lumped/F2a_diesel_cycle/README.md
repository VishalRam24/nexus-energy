# EC111 -- Diesel Generator -- F2a Diesel Cycle Model

## Model Description

Air-standard diesel cycle thermodynamic model with ODE-based dynamic governor/generator simulation. The model captures the four processes of the ideal diesel cycle, computes thermal efficiency from the compression and cutoff ratios, and simulates transient speed-load dynamics using a PI governor with fuel actuator dynamics.

## Physics

**Diesel Cycle (Air-Standard):**
```
1 -> 2  Isentropic compression:       T2 = T1 * r_c^(gamma-1)
2 -> 3  Constant-pressure heat add:   T3 = T2 * r_co
3 -> 4  Isentropic expansion:         T4 = T3 / (V4/V3)^(gamma-1)
4 -> 1  Constant-volume heat reject:  Q_rej = cv * (T4 - T1)
```

**Thermal Efficiency:**
```
eta_diesel = 1 - (1 / r_c^(gamma-1)) * ((r_co^gamma - 1) / (gamma * (r_co - 1)))
```

**Dynamic ODE (Governor + Rotational Dynamics):**
```
d(omega)/dt = (T_engine - T_gen - b*omega) / J
d(x_fuel)/dt = (x_fuel_cmd - x_fuel) / tau_act
d(int_err)/dt = omega_ref - omega   (PI governor integrator)
```

## Parameters

| Parameter | Value | Unit | Description |
|-----------|-------|------|-------------|
| P_rated | 500 | kW | Rated electrical output |
| r_c | 18 | -- | Compression ratio |
| gamma | 1.35 | -- | Specific heat ratio |
| n_cylinders | 6 | -- | Number of cylinders |
| V_displaced | 15 | L | Engine displacement |
| rpm_nominal | 1500 | rpm | Nominal speed (50 Hz) |
| J | 15 | kg*m^2 | Rotational inertia |
| eta_gen_rated | 0.96 | -- | Generator efficiency at rated |
| BSFC_rated | 210 | g/kWh | Brake specific fuel consumption |
| LHV_diesel | 42.7 | MJ/kg | Lower heating value |

## Inputs

| Name | Unit | Range | Description |
|------|------|-------|-------------|
| P_load | W | 0 -- 550,000 | Electrical load demand |
| load_fraction | -- | 0 -- 1.1 | Load as fraction of rated |
| dt | s | 0.001 -- 1.0 | Simulation time step |
| duration_s | s | 1 -- 300 | Simulation duration |

## Outputs

| Name | Unit | Description |
|------|------|-------------|
| t | s | Time array |
| omega_rpm | rpm | Engine speed |
| frequency_Hz | Hz | Electrical frequency |
| P_elec_W | W | Electrical power output |
| P_engine_W | W | Mechanical engine power |
| fuel_rate_kg_s | kg/s | Fuel consumption rate |
| eta_overall | -- | Overall efficiency |
| eta_thermal | -- | Diesel cycle thermal efficiency |
| BSFC_g_per_kWh | g/kWh | Brake specific fuel consumption |

## Accuracy

- Thermal efficiency: matches analytical formula to machine precision
- First law consistency: q_add - q_rej = w_net verified
- BSFC at rated: ~210 g/kWh (typical for medium-speed diesel genset)
- Dynamic simulation: governor recovers speed within 2% after load step

## Limitations

- Air-standard cycle (idealized, no combustion chemistry)
- No turbocharger dynamics
- No exhaust aftertreatment modeling
- Simplified fuel injection (no injection timing effects)
- No cylinder-to-cylinder variation

## Source

Heywood, J.B. (2018). Internal Combustion Engine Fundamentals, 2nd ed. McGraw-Hill.
