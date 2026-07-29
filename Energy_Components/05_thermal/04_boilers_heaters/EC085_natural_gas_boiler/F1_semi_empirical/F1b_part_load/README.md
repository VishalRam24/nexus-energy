# EC085 — Natural Gas Boiler — F1b Part-Load

## Model

Extends F1a with:

1. **Quadratic part-load curve**: eta(PLR) = a0 + a1*PLR + a2*PLR^2
   - Default: a0=0.75, a1=0.45, a2=-0.25 (peak ~0.95 at PLR~0.9)
2. **Flue gas losses**: Q_flue = m_flue * cp * (T_flue - T_air)
   - T_flue scales with PLR
3. **Standby loss**: Q_standby = 0.005 * Q_rated

## Inputs

| Parameter | Unit | Range | Default |
|-----------|------|-------|---------|
| PLR | - | 0-1 | - |
| flue_gas_temp | degC | 50-250 | auto |

## Outputs

| Parameter | Unit |
|-----------|------|
| efficiency | - |
| heat_output_kw | kW |
| fuel_input_kw | kW |
| flue_loss_kw | kW |
| standby_loss_kw | kW |

## References

- EnergyPlus Engineering Reference (2023), Boiler:HotWater
- EN 15502
- Stafford (2009), Energy and Buildings, 41(2), 168-175
