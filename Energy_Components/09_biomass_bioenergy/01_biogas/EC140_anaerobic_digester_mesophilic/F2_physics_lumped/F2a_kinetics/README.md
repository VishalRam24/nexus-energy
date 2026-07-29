# EC140 -- Anaerobic Digester (Mesophilic) -- F2a Monod Kinetics Model

## Model Description

Simplified ADM1 (Anaerobic Digestion Model No. 1) with Monod kinetics for substrate degradation and biomass growth. Models a mesophilic CSTR digester processing dairy manure. Includes Arrhenius temperature correction and empirical pH inhibition.

## Physics

**State Equations:**
```
dS/dt = (S_in - S)/HRT - (mu_max * S/(K_s + S)) * X / Y_xs
dX/dt = (mu_max * S/(K_s + S)) * X - k_d * X - X/HRT
```

**Monod Kinetics:**
```
mu = mu_max * S / (K_s + S)
```

**Temperature Correction (Arrhenius):**
```
f_T = exp(E_a/R * (1/T_ref - 1/T))
```

**pH Inhibition:**
```
f_pH = exp(-3 * ((pH - pH_opt) / (pH_high - pH_low))^2)
```

**Methane Production:**
```
V_ch4 = Y_ch4 * mu * X / Y_xs * V_reactor   [L/d at STP]
```

## Parameters

| Parameter | Value | Unit | Description |
|-----------|-------|------|-------------|
| V_reactor | 1000 | m^3 | Reactor volume |
| HRT | 20 | d | Hydraulic retention time |
| S_in | 40 | gCOD/L | Influent substrate (dairy manure) |
| mu_max | 0.4 | 1/d | Max specific growth rate (35 C) |
| K_s | 3.0 | gCOD/L | Half-saturation constant |
| Y_xs | 0.1 | gVSS/gCOD | Biomass yield |
| k_d | 0.02 | 1/d | Endogenous decay rate |
| Y_ch4 | 0.35 | L_CH4/gCOD | Methane yield per COD removed |
| T_ref | 35 | C | Reference temperature |
| E_a | 50000 | J/mol | Activation energy |

## Inputs

| Name | Unit | Range | Description |
|------|------|-------|-------------|
| S_in | gCOD/L | 0 -- 100 | Influent substrate concentration |
| HRT | d | 5 -- 60 | Hydraulic retention time |
| T | K | 293 -- 328 | Operating temperature |
| pH | -- | 5.5 -- 9.0 | Operating pH |
| dt | d | 0.01 -- 1.0 | Output time step |
| duration_d | d | 1 -- 365 | Simulation duration |

## Outputs

| Name | Unit | Description |
|------|------|-------------|
| t | d | Time array |
| S | gCOD/L | Substrate concentration |
| X | gVSS/L | Biomass concentration |
| V_ch4_rate_L_d | L/d | Methane production rate |
| V_ch4_cumulative_L | L | Cumulative methane produced |
| COD_removal_pct | % | COD removal efficiency |
| mu_eff | 1/d | Effective specific growth rate |

## Accuracy

- Monod saturation: verified at S=0 (mu=0) and S>>K_s (mu->mu_max)
- Arrhenius: f_T=1.0 at reference temperature (verified to machine precision)
- Mass balance: substrate consumed matches biomass produced / COD removed
- Steady-state: dynamic simulation converges to analytical solution within 10%

## Limitations

- Single-substrate, single-biomass (simplified vs. full ADM1 with ~26 state variables)
- No volatile fatty acid (VFA) speciation
- No gas-liquid mass transfer (assumes all CH4 exits immediately)
- No ammonia inhibition
- Fixed volume (no sludge removal dynamics)
- pH as external input (no internal pH calculation from VFA/alkalinity)

## Source

Batstone, D.J. et al. (2002). IWA Anaerobic Digestion Model No. 1 (ADM1). IWA Publishing.
Rittmann, B.E. & McCarty, P.L. (2001). Environmental Biotechnology. McGraw-Hill.
