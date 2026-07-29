# EC069 — Ground-Source Heat Pump (GSHP) — F1a COP Map

## Overview
COP map model for a ground-source heat pump using the Carnot fraction approach.
Shares the same architecture as EC068 (ASHP) but calibrated to GSHP rating conditions
(G10/W35: ground 10°C in, water 35°C out).

The key seasonal advantage of GSHP over ASHP is that the ground maintains a stable temperature
(~10°C in temperate climates) throughout winter, whereas air-source units operate at -10 to 5°C.
This eliminates defrost cycles and reduces compressor stress.

## Model Card

| Property | Value |
|---|---|
| EC ID | EC069 |
| Fidelity | F1a |
| Rated capacity | 15 kW_th |
| Rated COP | 4.5 at G10/W35 |
| Carnot fraction (η) | 0.365 (calibrated to COP=4.5 at G10/W35) |
| T_source range | 0 – 20 °C (ground/brine loop) |
| T_sink range | 25 – 65 °C (heating distribution) |

## Inputs / Outputs

| Input | Unit | Range | Description |
|---|---|---|---|
| T_source | °C | 0 – 20 | Ground loop supply temperature |
| T_sink | °C | 25 – 65 | Heating distribution temperature |
| part_load_ratio | - | 0 – 1 | Part-load ratio (default=1.0) |

| Output | Unit | Description |
|---|---|---|
| cop | - | Heating COP |
| heating_capacity_kw | kW | Thermal output |
| electrical_input_kw | kW | Electrical consumption |

## Physics

```
COP_Carnot = T_sink_K / (T_sink_K - T_source_K)
COP = η_Carnot × COP_Carnot   (η_Carnot = 0.365)
Q_heating = COP × W_compressor
W_elec = Q_rated × PLR / COP + W_aux
```

## Calibration Note

The Carnot fraction η=0.365 is derived from the GSHP rating point G10/W35:
```
COP_Carnot(10°C, 35°C) = (35+273.15)/(35-10) = 12.33
η = 4.5 / 12.33 = 0.365
```

Although the nominal η (0.50) claimed in the specification would yield COP=6.16 at G10/W35,
calibration to the actual rated COP of 4.5 requires η=0.365. This is consistent with
real GSHP literature which reports seasonal COP of 3.5–5.0 depending on ground temperature
and distribution system.

## Tests (12/12 passing)
- Output key completeness
- COP > 1 at all valid conditions
- COP at G10/W35 in [4.0, 5.5]
- COP > 4 at G10/W35
- GSHP seasonal advantage over ASHP at cold winter conditions
- COP decreases with temperature lift
- COP increases with source temperature
- Heating capacity = rated at PLR=1
- Benchmark: 1000 predictions < 1 second

## Data Sources
- Staffell, I. et al. (2012). "A review of domestic heat pumps." _Energy Environ. Sci._, 5, 9291–9306.
- ASHRAE Handbook — HVAC Applications (2019), Chapter 34: Ground-Source Heat Pumps.
- EN 15450 / ISO 13256-2: Rating conditions for ground-source heat pumps.

## Known Limitations
- Steady-state model; no ground thermal depletion or soil recovery dynamics
- Ground temperature assumed constant (single T_source input)
- No part-load efficiency curve (PLR affects capacity linearly, not COP)
- No frosting, defrost, or refrigerant cycling effects (not relevant for GSHP)
