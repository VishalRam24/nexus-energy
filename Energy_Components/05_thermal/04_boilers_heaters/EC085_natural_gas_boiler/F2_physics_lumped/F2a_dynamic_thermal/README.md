# EC085 -- Natural Gas Boiler -- F2a Dynamic Thermal

## Model Card

| Field | Value |
|-------|-------|
| Component | Natural Gas Boiler |
| EC ID | EC085 |
| Fidelity | F2a -- Dynamic Thermal Mass ODE |
| Version | 1.0.0 |

## Physics

Lumped thermal mass ODE for boiler body (water + metal):

```
(M_w*cp_w + M_body*cp_body) * dT_w/dt = Q_burner*eta_comb*(1-flue_loss)
                                          - m_dot*cp*(T_w - T_in)
                                          - UA_loss*(T_w - T_amb)
```

Features:
- Burner on/off cycling with deadband controller
- Part-load modulation (20-100%)
- Load-dependent combustion efficiency
- Flue gas losses (5% of burner input)
- Standby heat loss to ambient

## Default Unit: 500 kW Commercial Condensing Boiler
- Water mass: 500 kg, Body mass: 200 kg (steel)
- Combustion efficiency: 88-92% (part to full load)
- Design flow: 6 kg/s, Setpoint: 80 C

## Inputs

| Parameter | Unit | Default | Range |
|-----------|------|---------|-------|
| T_init_K | K | 293.15 | [273, 373] |
| T_in_K | K | 333.15 | [283, 363] |
| m_dot | kg/s | 6.0 | [0, 12] |
| T_set_K | K | 353.15 | [313, 373] |
| dt | s | 1.0 | [0.1, 60] |
| duration_s | s | 600 | [1, 86400] |

## Outputs

| Output | Unit | Description |
|--------|------|-------------|
| T_boiler | K | Boiler water temperature |
| modulation | - | Burner modulation (0-1) |
| Q_burner_W | W | Burner heat input |
| Q_output_W | W | Useful heat output to water |
| Q_loss_W | W | Standby/jacket losses |
| thermal_efficiency | - | Q_output / Q_fuel |

## Limitations
- Well-mixed boiler (single temperature node)
- No detailed flue gas composition model
- Simple proportional burner controller (no PID)
- Constant water properties

## References
- Rasmussen (2012), Dynamic Modelling of Boilers
- EN 15502 Condensing Boiler Standard
