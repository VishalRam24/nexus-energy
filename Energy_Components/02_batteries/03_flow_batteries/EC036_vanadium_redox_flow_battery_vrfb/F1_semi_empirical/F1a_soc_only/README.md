# EC036 — Vanadium Redox Flow Battery (VRFB) — F1a Nernst + Ohmic

## Model Overview

Single-cell voltage from the Nernst equation (both vanadium couples) plus ohmic drop,
scaled to a full stack.

```
E_Nernst = E0 + 2*(R*T)/(n*F) * ln(SOC / (1-SOC))
V_cell   = E_Nernst - I * R_cell          (R_cell = R_area / A_electrode)
V_stack  = N_cells * V_cell

eta_V    = V_discharge(I) / V_charge(I)   (voltage efficiency)
```

## Parameters

| Parameter | Value | Unit | Note |
|-----------|-------|------|------|
| N_cells | 40 | – | Cells in series |
| A_electrode | 600 | cm2 | Active electrode area |
| E0 | 1.26 | V | Standard cell potential |
| R_area | 0.8 | Ohm.cm2 | Area-specific resistance |
| T | 298 | K | Operating temperature |

## Inputs / Outputs

**Inputs:**

| Name | Unit | Range |
|------|------|-------|
| soc | – | 0.1 – 0.9 |
| current | A | –100 to 100 (positive = discharge) |

**Outputs:**

| Name | Unit |
|------|------|
| cell_voltage  | V |
| stack_voltage | V |
| power         | W (positive = discharge) |
| efficiency    | – (voltage efficiency, valid for I>0) |

## Usage

```python
from scripts.predict import ComponentModel
model = ComponentModel()
result = model.predict({"soc": 0.5, "current": 50.0})
print(result)
```

## Reference

Blanc, C., Rufer, A. (2010). *Multiphysics and Energetic Modeling of a Vanadium Redox Flow Battery.*
In: Paths to Sustainable Energy, InTech, ch. 16.

## Limitations

- Crossover (vanadium ion diffusion across membrane) and self-discharge are not included (F1b adds these).
- Temperature dependence of E0 and R_cell is not modeled here.
- Electrolyte flow effects (pressure drop, pump power) are excluded.
