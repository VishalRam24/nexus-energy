# EC195 — Ammonia Synthesis (Haber-Bosch) — F1a Per-Pass Conversion Model

## Overview
Per-pass conversion model for the iron-catalyzed Haber-Bosch process. Conversion is computed as an Arrhenius-pressure function, limited by the Temkin-Pyzhev equilibrium constraint. Models the fundamental trade-off between kinetics (higher T increases rate) and thermodynamics (higher T decreases equilibrium conversion).

## Reaction
```
N2 + 3H2 → 2NH3    ΔH = -92 kJ/mol N2  (exothermic)
```

## Model Equations
```
X_calc = X_ref * (P/P_ref)^0.5 * exp(-Ea/R * (1/T - 1/T_ref))
X_eq   = Temkin-Pyzhev equilibrium limit: log10(K_p) = 2250.3/T - 0.8534 - 1.51e-4*T - 25.89
X      = min(X_calc, X_eq)

NH3_rate   = 2 * X * n_N2 * MW_NH3   [kg/s]
E_specific = E_design * (X_ref/X)^0.2  [GJ/tNH3]
```
Design parameters: T_ref=450°C, P_ref=200 bar, X_ref=0.15, Ea/R=8000 K

## Inputs
| Name | Unit | Range | Description |
|------|------|-------|-------------|
| temperature | degC | [350, 550] | Synthesis loop temperature |
| pressure | bar | [100, 300] | Synthesis loop pressure |
| n_n2_in | mol/s | — | N2 feed rate (default 1.0) |

## Outputs
| Name | Unit | Description |
|------|------|-------------|
| conversion_per_pass | - | Per-pass N2 conversion |
| nh3_rate_kgs | kg/s | NH3 production rate |
| energy_gj_per_ton | GJ/tNH3 | Specific energy consumption |
| efficiency | - | NH3 LHV / H2 LHV ratio |

## Design Point (T=450°C, P=200 bar, 1 mol/s N2)
| Parameter | Value |
|-----------|-------|
| Conversion per pass | ~0.15 |
| NH3 rate | ~0.0051 kg/s (per mol/s N2) |
| Specific energy | ~28 GJ/tNH3 |
| Efficiency | ~0.55 |

## Sources
1. Appl, M. (2011). Ammonia. In *Ullmann's Encyclopedia of Industrial Chemistry*. Wiley-VCH. DOI: 10.1002/14356007.a02_143

## Limitations
- Per-pass conversion model; recycle loop not explicitly simulated
- Temkin-Pyzhev equilibrium is approximate (±5% error vs rigorous thermodynamics)
- Catalyst poisoning, activation, and aging not included
- N2:H2 = 1:3 stoichiometric feed assumed; off-ratio feeds not modeled
