# EC008 — PEM Electrolyser (PEMEL) — F1a V-I Characteristic

## Model Card

| Field | Value |
|---|---|
| Component ID | EC008 |
| Component | PEM Electrolyser (PEMEL) |
| Fidelity | F1a — Semi-empirical V-I Characteristic |
| Version | 1.0.0 |
| Source | Garcia-Valverde et al. (2012), *Int. J. Hydrogen Energy*, 37(2), 1927–1938 |

## Physics

The cell voltage is decomposed into three contributions:

```
V_cell = E_rev + V_act + V_ohm
```

| Term | Equation | Description |
|---|---|---|
| E_rev | 1.229 − 0.0009·(T − 298) | Reversible (Nernst) voltage [V] |
| V_act | (R·T)/(α·n·F) · arcsinh(j / 2j₀) | Butler-Volmer activation overpotential [V] |
| V_ohm | j · R_membrane | Ohmic loss through Nafion membrane [V] |

Stack voltage: `V_stack = N_cells · V_cell`

Hydrogen rate (Faraday's law): `ṅ_H₂ = (j · A) / (2F)` [mol/s]

Efficiency (HHV basis): `η = (ṅ_H₂ · HHV_H₂) / P_stack`

## Default Parameters

| Parameter | Value | Unit | Notes |
|---|---|---|---|
| T | 353.15 | K | 80 °C operating temperature |
| N_cells | 20 | — | Stack size |
| electrode_area | 100 | cm² | Active area |
| j0 | 1×10⁻⁴ | A/cm² | Exchange current density |
| alpha | 0.5 | — | Charge transfer coefficient |
| R_membrane | 0.2 | Ω·cm² | Nafion 117 resistance |

## Inputs / Outputs

**Inputs:**
- `current_density` — A/cm², range [0, 2.0]
- `temperature` — °C, range [40, 90]

**Outputs:**
- `cell_voltage_V` — cell voltage [V]
- `stack_voltage_V` — stack voltage [V]
- `hydrogen_rate_mol_s` — H₂ production rate [mol/s]
- `power_W` — electrical power input [W]
- `efficiency` — HHV efficiency [−]
- `E_rev_V`, `V_act_V`, `V_ohm_V` — voltage component breakdown [V]

## Usage

```python
from scripts.predict import ComponentModel

model = ComponentModel()
out = model.predict({"current_density": 1.0, "temperature": 80.0})
print(out)
```

## File Structure

```
F1a_vi_characteristic/
├── scripts/
│   ├── model.py        # Physics equations
│   ├── predict.py      # ComponentModel wrapper
│   ├── test_model.py   # Test suite (12 tests)
│   └── simulate.py     # Plotly HTML report generator
├── data/
│   └── parameters.json # Default parameters with metadata
├── model_files/
│   └── EC008_vi_characteristic_report.html  (generated)
└── README.md
```

## Physics Validity

- V_cell > E_rev for all j > 0 (overpotentials are strictly positive)
- V_cell increases monotonically with current density
- H₂ rate is strictly proportional to j (Faraday's law, linear)
- Efficiency < 1.0 for all operating points (thermodynamic limit)
- Higher temperature reduces cell voltage (lower activation + lower E_rev)
