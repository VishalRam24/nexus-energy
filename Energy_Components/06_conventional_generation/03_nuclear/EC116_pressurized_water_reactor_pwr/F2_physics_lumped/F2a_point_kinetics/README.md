# EC116 -- Pressurized Water Reactor (PWR) -- F2a Point Kinetics Model

## Model Description

Six-group delayed neutron point kinetics coupled with lumped fuel and moderator thermal-hydraulic feedback. This is a stiff ODE system requiring implicit solvers (Radau or BDF). The model captures the essential reactor dynamics including prompt jump, delayed neutron effects, Doppler broadening feedback, and moderator temperature feedback.

## Physics

**Neutron Kinetics (6-group delayed neutrons):**
```
dn/dt = (rho - beta) / Lambda * n + sum_i(lambda_i * C_i)
dC_i/dt = beta_i / Lambda * n - lambda_i * C_i   (i = 1..6)
```

**Temperature Feedback:**
```
rho = rho_ext + alpha_f * (T_f - T_f0) + alpha_m * (T_m - T_m0)
```

**Thermal-Hydraulics (Lumped):**
```
dT_f/dt = (P - hA_fg * (T_f - T_m)) / (m_f * cp_f)
dT_m/dt = (hA_fg * (T_f - T_m) - m_dot * cp * (T_m - T_in)) / (m_m * cp_m)
```

**Total: 9 coupled ODEs** (1 neutron + 6 precursor + 2 temperature)

## Parameters

| Parameter | Value | Unit | Description |
|-----------|-------|------|-------------|
| P_thermal | 3000 | MW | Rated thermal power |
| Lambda | 2e-5 | s | Prompt neutron generation time |
| beta_total | 0.006502 | -- | Total delayed neutron fraction |
| alpha_f | -2.5e-5 | dk/K | Fuel Doppler coefficient |
| alpha_m | -1.5e-4 | dk/K | Moderator temperature coefficient |
| T_f0 | 900 | K | Reference fuel temperature |
| T_m0 | 580 | K | Reference moderator temperature |
| T_inlet | 565 | K | Coolant inlet temperature |

## Delayed Neutron Data (U-235)

| Group | beta_i | lambda_i [1/s] | Half-life [s] |
|-------|--------|----------------|---------------|
| 1 | 0.000215 | 0.0124 | 55.9 |
| 2 | 0.001424 | 0.0305 | 22.7 |
| 3 | 0.001274 | 0.111 | 6.24 |
| 4 | 0.002568 | 0.301 | 2.30 |
| 5 | 0.000748 | 1.14 | 0.608 |
| 6 | 0.000273 | 3.01 | 0.230 |

## Stiffness Note

This system is inherently stiff due to the wide spread of time constants:
- Prompt neutron lifetime: ~20 microseconds
- Shortest precursor half-life: ~0.23 seconds
- Longest precursor half-life: ~56 seconds
- Thermal time constants: ~seconds

**Must use implicit solver:** `method='Radau'` or `method='BDF'` with `rtol=1e-8`.

## Inputs

| Name | Unit | Range | Description |
|------|------|-------|-------------|
| rho_ext | dk/k | -0.01 -- 0.005 | External reactivity (control rod) |
| dt | s | 0.001 -- 1.0 | Output time step |
| duration_s | s | 0.01 -- 1000 | Simulation duration |

## Outputs

| Name | Unit | Description |
|------|------|-------------|
| t | s | Time array |
| n | -- | Normalized neutron population |
| C | -- | 6 precursor concentrations (6 x N array) |
| T_f | K | Fuel temperature |
| T_m | K | Moderator temperature |
| P_thermal_W | W | Thermal power |
| P_elec_W | W | Electrical power |
| rho | dk/k | Total reactivity |

## Accuracy

- Equilibrium initial conditions: verified to machine precision
- Zero-reactivity stability: n drift < 0.001 over 50s
- Prompt jump approximation: matches theory within 5%
- Radau and BDF solvers agree within 0.1%

## Limitations

- Lumped (0D) model: no spatial flux distribution
- Single fuel and moderator temperature node
- No xenon/samarium poisoning dynamics
- No boron dilution modeling
- Linear temperature feedback coefficients (valid near reference)

## Source

Duderstadt, J.J. & Hamilton, L.J. (1976). Nuclear Reactor Analysis. Wiley.
Stacey, W.M. (2007). Nuclear Reactor Physics, 2nd ed. Wiley-VCH.
Keepin, G.R. (1965). Physics of Nuclear Kinetics. Addison-Wesley.
