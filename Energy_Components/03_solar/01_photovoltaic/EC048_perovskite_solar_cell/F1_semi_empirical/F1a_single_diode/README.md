# EC048 — Perovskite Solar Cell — F1a Single-Diode Model

## Overview
Single-diode (De Soto 5-parameter) model adapted for lab-scale MAPbI3 perovskite solar cells.
Uses the same framework as EC044 (mono-Si) but with perovskite-specific parameters reflecting
the wider bandgap (1.55 eV), higher Voc, and trap-assisted recombination (n=1.5).

## Model Card

| Property | Value |
|---|---|
| EC ID | EC048 |
| Fidelity | F1a |
| Library | pvlib (BSD-3, `singlediode` solver via Lambert-W) |
| Cell type | MAPbI3 perovskite (lab-scale, 25 cm²) |
| STC Pmp | 0.5 W (Vmp=1.0 V × Imp=0.5 A) |
| STC Voc | 1.18 V |
| STC Isc | 0.628 A (25.1 mA/cm²) |
| Efficiency | 20% at STC |
| Bandgap | 1.55 eV (vs 1.12 eV for Si) |
| Ideality factor | 1.5 (trap-assisted recombination) |

## Inputs / Outputs

| Input | Unit | Range | Description |
|---|---|---|---|
| irradiance | W/m² | 0 – 1200 | Incident irradiance on cell |
| cell_temperature | °C | -10 – 80 | Cell junction temperature |

| Output | Unit | Description |
|---|---|---|
| v_mp | V | Voltage at maximum power point |
| i_mp | A | Current at maximum power point |
| p_mp | W | Maximum power output |
| v_oc | V | Open-circuit voltage |
| i_sc | A | Short-circuit current |
| efficiency | - | Cell efficiency = p_mp / (G × area) |

## Physics

The De Soto 5-parameter single-diode equation:

```
I = I_L - I_o * [exp((V + I*Rs) / a) - 1] - (V + I*Rs) / R_sh
```

Parameters scale with irradiance (G) and temperature (T):
- `I_L ∝ G` (+ small temp correction)
- `I_o ∝ T³ × exp(Eg / kT)` (Boltzmann-weighted bandgap)
- `a = n × Ns × k × T / q` (ideality factor n=1.5 for perovskite)
- `R_sh ∝ 1/G`

## Perovskite vs Si Differences

| Parameter | Perovskite (EC048) | Si (EC044) |
|---|---|---|
| Bandgap | 1.55 eV | 1.12 eV |
| Voc (STC) | 1.18 V | 38.3 V (60-cell module) |
| Jsc | 25.1 mA/cm² | 9.39 A (module) |
| n_diode | 1.5 | 1.0 |
| Temp coeff (Voc) | −3 mV/K | −3 mV/K |

## De Soto 5-Parameter Values (STC, fitted to specs)

| Parameter | Value | Unit |
|---|---|---|
| I_L_ref | 0.6347 | A |
| I_o_ref | 2.525×10⁻¹⁴ | A |
| R_s | 0.0938 | Ω |
| R_sh_ref | 8.81 | Ω |
| a_ref | 0.03854 | V |
| EgRef | 1.55 | eV |
| dEgdT | −4.5×10⁻⁴ | eV/K |

## Tests (12/12 passing)
- Output key completeness
- STC power ~0.5 W, Voc > 1.0 V, efficiency 20%
- Zero irradiance → zero output
- Power proportional to G (monotone)
- Power decreases with temperature (negative temp coeff)
- Voc > Vmp, Isc ≥ Imp always
- Benchmark: 1000 predictions < 1 second

## Data Sources
- De Soto, W. et al. (2006). "Improvement and validation of a model for photovoltaic array performance." _Solar Energy_, 80(1), 78–88.
- Miyano, K. et al. (2016). "Lead halide perovskite photovoltaic as a model p-i-n diode." _J. Phys. Chem. Lett._, 7, 2199–2202.
- NREL Best Research-Cell Efficiency Chart (2024).

## Known Limitations
- Stability and hysteresis not modeled (F1a is steady-state)
- Parameters fitted to ideal lab-cell; real commercial modules vary significantly
- No degradation, moisture, or light-soaking effects (handled in higher fidelity models)
- 5-parameter De Soto is designed for Si; fit quality may differ for perovskite
