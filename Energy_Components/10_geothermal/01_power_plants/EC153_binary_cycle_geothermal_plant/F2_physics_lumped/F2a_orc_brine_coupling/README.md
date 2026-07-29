# EC153 -- Binary Cycle Geothermal Plant -- F2a ORC-Brine Coupling

## Model Description
Physics-lumped ODE model of a binary cycle geothermal power plant coupling an Organic Rankine Cycle (ORC) with geothermal brine heat exchange.

## Physics
- **Brine side:** Energy balance with pinch-point constraint; optional long-term temperature decline
- **ORC side:** Isobutane (R600a) cycle: pump -> preheater -> evaporator -> turbine -> condenser
- **Working fluid properties:** Polynomial fits from NIST data (no CoolProp dependency)
- **Thermal ODE:** Evaporator temperature dynamics with thermal inertia
- **Parasitic loads:** Working fluid pump, brine circulation pump

## Inputs
| Parameter | Unit | Default | Range |
|-----------|------|---------|-------|
| T_brine_in | K | 443.15 | 373-523 |
| T_evap_init | K | 393.15 | 313-423 |
| dt | s | 1.0 | - |
| duration_s | s | 300.0 | 1-86400 |
| brine_decline_years | years | 0 | 0-30 |

## Outputs
| Variable | Unit | Description |
|----------|------|-------------|
| W_net | W | Net electrical power output |
| W_turbine | W | Gross turbine power |
| W_parasitic | W | Total parasitic pump power |
| Q_in | W | Heat input to ORC |
| eta_thermal | - | Net thermal efficiency |
| T_evap | K | Evaporator temperature |
| T_brine_out | K | Brine reinjection temperature |

## References
- DiPippo (2012), Geothermal Power Plants, 3rd ed., Elsevier
- Franco & Villani (2009), Optimal design of binary cycle power plants
- NIST WebBook for isobutane thermodynamic properties

## Limitations
- Polynomial fluid property fits valid ~280-430 K (not supercritical)
- Lumped thermal model (no spatial resolution in heat exchangers)
- Single working fluid (isobutane); no mixture optimization
