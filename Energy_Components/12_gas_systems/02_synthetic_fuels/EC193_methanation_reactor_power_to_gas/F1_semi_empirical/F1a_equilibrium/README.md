# EC193 — Methanation Reactor (Power-to-Gas) — F1a Sabatier Equilibrium Model

## Overview
Semi-empirical equilibrium conversion model for the catalytic Sabatier reaction used in power-to-gas (P2G) systems. Conversion is modeled as a Gaussian function of temperature (exothermic reaction, optimal at 300°C) multiplied by a pressure enhancement term.

## Reaction
```
CO2 + 4H2 → CH4 + 2H2O    ΔH = -165 kJ/mol  (exothermic)
```

## Model Equations
```
X = X_max * exp(-k_T * ((T - T_opt) / T_opt)^2) * (P / P_ref)^0.1
X = min(X, H2_CO2_ratio / 4)          [H2 availability limit]

CH4_rate   = X * n_CO2_in              [mol/s]
H2_consumed = 4 * X * n_CO2_in        [mol/s]
eta        = X * LHV_CH4 / (4 * LHV_H2)
Q_heat     = X * n_CO2_in * DH_rxn    [kW]
```
Parameters: X_max=0.98, T_opt=300°C, P_ref=10 bar, k_T=10.0

## Inputs
| Name | Unit | Range | Description |
|------|------|-------|-------------|
| temperature | degC | [200, 500] | Reactor temperature |
| pressure | bar | [1, 30] | Reactor pressure |
| h2_co2_ratio | mol/mol | [3.5, 5.0] | H2/CO2 feed ratio |
| n_co2_in | mol/s | — | CO2 feed rate (default 1.0) |

## Outputs
| Name | Unit | Description |
|------|------|-------------|
| conversion | - | CO2-to-CH4 conversion fraction |
| ch4_rate_mols | mol/s | CH4 production rate |
| efficiency | - | Energy efficiency X*LHV_CH4/(4*LHV_H2) |
| heat_released_kw | kW | Exothermic heat release |

## Design Point (T=300°C, P=10 bar, H2/CO2=4, n_CO2=1 mol/s)
| Parameter | Value |
|-----------|-------|
| Conversion X | 0.98 |
| CH4 rate | 0.98 mol/s |
| Efficiency | ~0.83 |
| Heat released | ~161 kW |

## Sources
1. Gao, J., Wang, Y., Ping, Y., Hu, D., Xu, G., Gu, F., Su, F. (2012). A thermodynamic analysis of methanation reactions of carbon oxides for the production of synthetic natural gas. *RSC Advances*, 2, 2358–2368.

## Limitations
- Empirical Gaussian fit; does not use rigorous chemical equilibrium constants
- Kinetics (approach to equilibrium) not included; assumes sufficient residence time
- Single-pass conversion only; no recycle loop
- Catalyst deactivation and coking not modeled
