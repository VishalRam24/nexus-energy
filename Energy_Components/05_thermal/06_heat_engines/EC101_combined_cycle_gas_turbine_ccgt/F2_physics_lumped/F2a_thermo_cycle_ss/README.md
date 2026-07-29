# EC101 -- Combined Cycle Gas Turbine (CCGT) -- F2a Thermo Cycle SS

## Model Card

| Field | Value |
|-------|-------|
| Component | Combined Cycle Gas Turbine (CCGT) |
| EC ID | EC101 |
| Fidelity | F2a -- Steady-State Thermodynamic Cycle (Brayton + Rankine) |
| Version | 1.0.0 |

## Physics

Combined Brayton (gas turbine) + Rankine (steam) cycle steady-state analysis.

**Brayton Cycle:**
- 1->2: Compressor with isentropic efficiency
- 2->3: Combustor with fuel LHV and pressure drop
- 3->4: Gas turbine with isentropic efficiency

**HRSG:** Exhaust heat recovery with effectiveness model.

**Rankine Cycle:**
- a->b: Feed pump
- b->c: HRSG steam generation
- c->d: Steam turbine with isentropic efficiency
- d->a: Condenser

Air treated as ideal gas with cp(T) polynomial. Steam/water with simple polynomial fits.

## Default Unit: 400 MW Class F-CCGT
- PR = 18, TIT = 1300 C, m_dot_air = 450 kg/s
- HP steam: 120 bar / 565 C
- Condenser: 0.05 bar
- Target combined efficiency: 55-62%

## Inputs

| Parameter | Unit | Default | Range |
|-----------|------|---------|-------|
| TIT_K | K | 1573.15 | [1273, 1873] |
| PR | - | 18 | [8, 35] |
| m_dot_air | kg/s | 450 | [100, 800] |
| load_fraction | - | 1.0 | [0.3, 1.0] |

## Outputs

| Output | Unit | Description |
|--------|------|-------------|
| W_total_MW | MW | Total electrical output |
| W_gt_elec_MW | MW | Gas turbine electrical output |
| W_st_elec_MW | MW | Steam turbine electrical output |
| eta_combined | - | Combined cycle efficiency |
| heat_rate_kJ_kWh | kJ/kWh | Heat rate |
| T_exhaust_K | K | Gas turbine exhaust temperature |

## Limitations
- Steady-state only (no dynamics)
- Simplified steam properties (no CoolProp)
- Single-pressure HRSG (simplified from 3-pressure)
- Simple part-load via TIT modulation only
- No ambient temperature corrections

## References
- Boyce (2012), Gas Turbine Engineering Handbook
- Kehlhofer et al. (2009), Combined-Cycle Gas & Steam Turbine Power Plants
