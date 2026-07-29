# EC085 — Natural Gas Boiler (Condensing) — F1a Part-Load Efficiency

## Model Card

| Field | Value |
|---|---|
| Component ID | EC085 |
| Component | Natural Gas Boiler (Condensing) |
| Fidelity | F1a — Part-Load Efficiency |
| Version | 1.0.0 |
| Source | EnergyPlus Engineering Reference (2023); Stafford (2009), *Energy and Buildings*, 41(2), 168–175 |

## Physics

### Part-Load Efficiency

```
eta(PLR) = eta_nom * (a0 + a1*PLR + a2*PLR^2)
```

Constraint: a0 + a1 + a2 = 1.0 so that eta(PLR=1) = eta_nom exactly.

### Thermal Output and Fuel Input

```
Q_out   = PLR * Q_rated              [kW]
Q_fuel  = Q_out / eta(PLR)           [kW thermal input from gas]
V_gas   = Q_fuel / LHV_kWh_m3       [m³/h]
```

### Condensing Temperature Correction

```
T <= 55°C:  factor = 1.05 - (T - 30) * 0.002     (condensing mode)
T > 55°C:   factor = 1.00 - (T - 55) * 0.0032    (non-condensing penalty)
eta_eff = eta(PLR) * factor
```

## Default Parameters

| Parameter | Value | Unit | Notes |
|---|---|---|---|
| Q_rated | 50 | kW | Rated thermal output |
| eta_nom | 0.95 | — | Full-load efficiency |
| a0 | 0.1 | — | Part-load polynomial constant |
| a1 | 0.9 | — | Part-load polynomial linear |
| a2 | 0.0 | — | Part-load polynomial quadratic |
| PLR_min | 0.1 | — | Minimum stable PLR |
| LHV_gas | 36.6 | MJ/m³ | Lower heating value |

## Inputs / Outputs

**Inputs:**
- `part_load_ratio` — PLR [0, 1]
- `supply_temp` — supply water temperature [°C], range [30, 80]

**Outputs:**
- `thermal_output_kw` — useful heat output [kW]
- `fuel_input_kw` — gas combustion thermal input [kW]
- `efficiency` — net efficiency (with condensing correction) [−]
- `gas_consumption_m3h` — volumetric gas flow [m³/h]
- `PLR_effective` — effective PLR (clamped to PLR_min) [−]
- `condensing_factor` — temperature correction factor [−]

## Usage

```python
from scripts.predict import ComponentModel

model = ComponentModel()
out = model.predict({"part_load_ratio": 0.6, "supply_temp": 55.0})
print(out)
```

## File Structure

```
F1a_part_load_eta/
├── scripts/
│   ├── model.py        # Physics equations
│   ├── predict.py      # ComponentModel wrapper
│   ├── test_model.py   # Test suite (12 tests)
│   └── simulate.py     # Plotly HTML report generator
├── data/
│   └── parameters.json
├── model_files/
│   └── EC085_boiler_report.html  (generated)
└── README.md
```

## Physics Validity

- eta ≤ 1.0 at all operating points
- Fuel input ≥ thermal output (energy conservation)
- Efficiency drops at low PLR (part-load penalty)
- Condensing boilers gain efficiency bonus at low supply temperatures
- Below PLR_min the model clamps to minimum stable operation
