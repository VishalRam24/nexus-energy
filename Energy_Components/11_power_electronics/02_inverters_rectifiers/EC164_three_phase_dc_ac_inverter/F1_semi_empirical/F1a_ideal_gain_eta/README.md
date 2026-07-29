# EC164 — Three-Phase DC-AC Inverter — F1a Ideal Gain + Efficiency

## Model Card

| Field | Value |
|---|---|
| Component ID | EC164 |
| Component | Three-Phase DC-AC Inverter |
| Fidelity | F1a — Ideal Gain + Part-Load Efficiency |
| Version | 1.0.0 |
| Source | Mohan et al. (2003), *Power Electronics*, 3rd ed. Wiley; IEC 61683 |

## Physics

### Voltage Gain (SVPWM)

```
V_ac_rms = m * V_dc / sqrt(2)    [V, line-to-line RMS]
```

Where m ∈ [0,1] is the space-vector modulation index.

### Part-Load Efficiency

```
PLR = P_load / P_rated
eta(PLR) = eta_rated - k1*(1 - PLR) - k2*(1 - PLR)^2
```

### Power Balance

```
P_in    = P_load / eta(PLR)
P_loss  = P_in - P_load
```

### AC Current

```
I_ac_rms = P_out / (sqrt(3) * V_ac_rms * PF)
```

## Default Parameters

| Parameter | Value | Unit | Notes |
|---|---|---|---|
| V_dc_rated | 800 | V | HV battery bus |
| P_rated | 100,000 | W | 100 kW inverter |
| eta_rated | 0.98 | — | Efficiency at full load |
| k1 | 0.02 | — | Part-load loss (linear) |
| k2 | 0.005 | — | Part-load loss (quadratic) |
| f_sw | 10,000 | Hz | IGBT switching frequency |

## Inputs / Outputs

**Inputs:**
- `v_dc` — DC bus voltage [V]
- `p_load` — output power [W], range [0, P_rated]
- `modulation_index` — SVPWM modulation index [0, 1]
- `power_factor` — load power factor [−], optional (default 1.0)

**Outputs:**
- `v_ac_rms_V` — line-to-line AC RMS voltage [V]
- `i_ac_rms_A` — line RMS current [A]
- `efficiency` — [−]
- `p_in_W` — DC power input [W]
- `p_loss_W` — total losses [W]
- `PLR` — part-load ratio [−]

## Usage

```python
from scripts.predict import ComponentModel

model = ComponentModel()
out = model.predict({
    "v_dc": 800.0,
    "p_load": 80000.0,
    "modulation_index": 0.9
})
print(out)
```

## Physics Validity

- V_ac_rms < V_dc for all m < sqrt(2) (always satisfied for m ≤ 1)
- eta ≤ 1.0 at all operating points
- P_loss > 0 for all P_load > 0
- Efficiency drops at low part-load ratio (iron/switching losses dominate)
