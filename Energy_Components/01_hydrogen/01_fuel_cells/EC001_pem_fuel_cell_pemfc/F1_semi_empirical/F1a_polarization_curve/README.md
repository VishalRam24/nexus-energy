# EC001 — PEM Fuel Cell (PEMFC) — F1a Polarization Curve

## Model Card

| Field | Value |
|---|---|
| Component ID | EC001 |
| Component | PEM Fuel Cell (PEMFC) |
| Fidelity | F1a — Polarization Curve (Amphlett semi-empirical) |
| Version | 1.0.0 |
| Source | Amphlett et al. (1995), *J. Electrochem. Soc.*, 142(1), 1–8 |

## Physics

The cell voltage is the Nernst voltage minus three loss terms:

```
V_cell = E_Nernst - V_act - V_ohm - V_conc
```

| Term | Equation | Description |
|---|---|---|
| E_Nernst | 1.229 − 8.5×10⁻⁴·(T−298.15) + RT/2F·ln(pH2·√pO2) | Open-circuit voltage [V] |
| V_act | −(ξ1 + ξ2T + ξ3T·ln(cO2) + ξ4T·ln(j)) | Activation overpotential [V] |
| V_ohm | j·t_mem/σ_mem | Ohmic loss (Nafion membrane) [V] |
| V_conc | −B·ln(1 − j/j_L) | Concentration/mass-transport loss [V] |

Amphlett coefficients: ξ1=−0.948, ξ2=2.86×10⁻³, ξ3=7.6×10⁻⁵, ξ4=−1.93×10⁻⁴

Membrane conductivity (Springer model):
`σ_mem = (0.005139λ − 0.00326) · exp(1268·(1/303 − 1/T))`

## Default Parameters

| Parameter | Value | Unit | Notes |
|---|---|---|---|
| T | 343.15 | K | 70 °C |
| N_cells | 40 | — | Stack size |
| electrode_area | 232 | cm² | Ballard Mark IV reference |
| pH2 | 1.0 | atm | Pure H2 |
| pO2 | 0.21 | atm | Air-fed |
| j_L | 1.5 | A/cm² | Limiting current density |
| t_mem | 0.0178 | cm | Nafion 117 (178 µm) |
| lambda_mem | 14.0 | — | Water content |

## Inputs / Outputs

**Inputs:**
- `current_density` — A/cm², range [0, j_L)
- `temperature` — °C, range [50, 90]

**Outputs:**
- `cell_voltage_V`, `stack_voltage_V` — [V]
- `power_density_W_cm2` — [W/cm²]
- `stack_power_W` — [W]
- `efficiency` — voltage efficiency vs. HHV (1.481 V) [−]
- `E_Nernst_V`, `V_act_V`, `V_ohm_V`, `V_conc_V` — voltage breakdown [V]

## Usage

```python
from scripts.predict import ComponentModel

model = ComponentModel()
out = model.predict({"current_density": 0.6, "temperature": 70.0})
print(out)
```

## File Structure

```
F1a_polarization_curve/
├── scripts/
│   ├── model.py        # Physics equations (Amphlett model)
│   ├── predict.py      # ComponentModel wrapper
│   ├── test_model.py   # Test suite (12 tests)
│   └── simulate.py     # Plotly HTML report generator
├── data/
│   └── parameters.json
├── model_files/
│   └── EC001_polarization_curve_report.html  (generated)
└── README.md
```

## Physics Validity

- V_cell decreases monotonically with current density
- V_cell < E_Nernst for all j > 0
- Power density has an interior maximum (bell-shaped curve)
- Concentration loss diverges as j → j_L
- Efficiency < 1.0 at all operating points
