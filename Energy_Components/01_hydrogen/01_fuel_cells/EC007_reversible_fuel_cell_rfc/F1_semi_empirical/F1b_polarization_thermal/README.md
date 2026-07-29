# EC007 Reversible Fuel Cell (RFC) — F1b Polarization-Thermal

## Overview
Temperature-dependent polarization model for an RFC operating in both FC (discharge) and electrolyser (charge) modes.

## F1b Additions over F1a
- Arrhenius exchange current density i0(T) for both ORR (FC) and OER (electrolyser)
- Nafion membrane conductivity sigma(T) via Springer 1991
- Temperature-dependent Nernst voltage
- Lumped thermal balance: dT/dt = (Q_gen - UA*(T - T_cool)) / (m*cp)

## Key Physics
- FC mode: V_cell = E_nernst - V_act(j,T) - V_ohm(j,T) - V_conc(j)
- EL mode: V_cell = E_nernst + V_act(j,T) + V_ohm(j,T) + V_conc(j)
- Heat: Q = j*(E_tn - V_cell) [FC], Q = j*(V_cell - E_tn) [EL]

## References
- Amphlett et al. (1995). J. Electrochem. Soc. 142(1), 1-8.
- Springer et al. (1991). J. Electrochem. Soc. 138(8), 2334-2342.
- Grigoriev et al. (2020). Int. J. Hydrogen Energy 45(53), 26651-26657.
