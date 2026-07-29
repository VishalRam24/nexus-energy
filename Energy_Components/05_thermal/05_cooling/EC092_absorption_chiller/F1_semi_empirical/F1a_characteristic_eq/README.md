# EC092 — Absorption Chiller — F1a Characteristic Equation

## Model Card

| Field | Value |
|-------|-------|
| Component | Absorption Chiller (Single-Effect LiBr-H2O) |
| EC ID | EC092 |
| Fidelity | F1a — Semi-Empirical Characteristic Equation |
| Path | `05_thermal/05_cooling/EC092_absorption_chiller/F1_semi_empirical/F1a_characteristic_eq/` |

## Model Equation

```
COP = COP_max * (1 - exp(-alpha * dT_driving / dT_ref))
dT_driving = T_generator - T_condenser

Q_generator  = Q_cool / COP          (heat input from waste heat / gas burner)
Q_reject     = Q_generator + Q_cool  (heat to cooling tower — First Law)
```

## Parameters

| Parameter | Value | Unit | Source |
|-----------|-------|------|--------|
| COP_max | 0.75 | — | Single-effect LiBr-H2O theoretical limit ~0.80 |
| alpha | 3.0 | — | Shape factor (steepness of COP rise) |
| dT_ref | 50.0 | K | Reference temperature difference |
| Q_cool_rated | 500 | kW | Rated cooling capacity |

## Inputs / Outputs

### Inputs
| Name | Unit | Range | Description |
|------|------|-------|-------------|
| T_generator | degC | 70–120 | Heat source temperature (generator) |
| T_condenser | degC | 25–45 | Cooling water temperature (condenser/absorber) |
| T_evaporator | degC | 4–15 | Chilled water supply temperature |
| Q_cool_kw | kW | 0–500 | Cooling demand (optional, default = rated) |

### Outputs
| Name | Unit | Description |
|------|------|-------------|
| cop | — | Coefficient of Performance (cooling) |
| cooling_kw | kW | Cooling delivered to chilled water loop |
| heat_input_kw | kW | Heat input required at generator |
| heat_rejection_kw | kW | Total heat rejected to cooling tower |

## Physics Checks (all pass)
- COP < 0.80 (single-effect LiBr-H2O thermodynamic limit)
- COP increases monotonically with T_generator
- COP decreases monotonically with T_condenser
- Q_reject = Q_generator + Q_cool (First Law, verified to machine precision)
- Q_reject > Q_cool at all conditions

## Rated Performance
At T_gen=90°C, T_cond=35°C: COP ≈ 0.70 (matches Herold et al. reference data)

## Benchmark
1000 predictions in < 1 ms (NumPy vectorized)

## Limitations
- Single-effect cycle only (COP_max < 0.80). Double-effect machines reach COP ~1.2–1.4.
- T_evaporator is accepted as input but does not influence COP in this sub-fidelity (F1a).
  Use F1b for evaporator temperature coupling.
- No part-load degradation model — COP is independent of load fraction at this fidelity.
- No crystallisation risk check (LiBr concentration limits not enforced).

## Data Sources
- Herold, K.E., Radermacher, R. & Klein, S.A. (2016). *Absorption Chillers and Heat Pumps*, 2nd ed. CRC Press.
- Gordon, J.M. & Ng, K.C. (2000). *Cool Thermodynamics*. Cambridge International Science.

## License
BSD-3
