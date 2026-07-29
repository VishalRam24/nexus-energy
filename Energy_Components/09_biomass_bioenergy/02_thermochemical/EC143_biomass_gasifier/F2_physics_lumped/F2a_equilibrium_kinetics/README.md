# EC143 -- Biomass Gasifier -- F2a Chemical Equilibrium Model

## Model Description

Physics-lumped equilibrium model for downdraft biomass gasification. Solves a nonlinear system
of equations (atom balances + equilibrium constraints) using `scipy.optimize.fsolve` to determine
the equilibrium syngas composition at a given temperature, equivalence ratio, and biomass composition.

## Physics

**Overall gasification reaction:**
```
CH_x O_y N_z + w*H2O + m*(O2 + 3.76*N2) -> n1*CO + n2*CO2 + n3*H2 + n4*H2O + n5*CH4 + n6*N2
```

**Equilibrium reactions:**
- Water-gas shift: CO + H2O <-> CO2 + H2, K_wgs(T) = exp(4577.8/T - 4.33)
- Methanation: C + 2H2 <-> CH4, K_meth(T) = exp(7082/T - 7.466 + 0.372*ln(T))

**Constraints:** C, H, O, N atom balances + 2 equilibrium equations.

## Inputs

| Parameter | Unit | Default | Range |
|-----------|------|---------|-------|
| biomass_C | mass fraction | 0.50 | 0.35-0.60 |
| biomass_H | mass fraction | 0.06 | 0.03-0.08 |
| biomass_O | mass fraction | 0.42 | 0.30-0.50 |
| biomass_N | mass fraction | 0.01 | 0.00-0.05 |
| moisture_content | - | 0.15 | 0.0-0.40 |
| equivalence_ratio | - | 0.30 | 0.15-0.50 |
| temperature_K | K | 1073.15 | 973-1373 |

## Outputs

| Parameter | Unit |
|-----------|------|
| composition_dry_mol_pct | mol% (CO, CO2, H2, CH4, N2) |
| composition_wet_mol_pct | mol% (includes H2O) |
| LHV_syngas_MJ_Nm3 | MJ/Nm3 |
| cold_gas_efficiency | - |
| H2_CO_ratio | - |
| gas_yield_Nm3_per_kg | Nm3/kg biomass |

## References

- Zainal et al. (2001) Energy Conversion & Management, 42, 1499-1515
- Li et al. (2004) Biomass & Bioenergy, 26, 171-193
- Jarungthammachote & Dutta (2007) Energy Conversion & Management, 48, 2718-2731

## Limitations

- Assumes thermodynamic equilibrium (actual gasifiers may not reach equilibrium)
- Does not model tar formation or char conversion kinetics
- Temperature is an input (not self-consistently determined from energy balance)
- No pressure effects beyond methanation equilibrium
