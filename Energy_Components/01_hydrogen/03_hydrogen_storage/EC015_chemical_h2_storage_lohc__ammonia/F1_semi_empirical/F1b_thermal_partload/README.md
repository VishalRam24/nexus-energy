# EC015 Chemical H2 Storage (LOHC/Ammonia) — F1b Thermal + Part-load

## F1b Additions over F1a
- Arrhenius temperature-dependent efficiency: eta(T) = eta_nom * exp(-Ea/R*(1/T - 1/T_nom))
- Part-load efficiency penalty: eta_pl = (F/F_nom)^n (power-law, n~0.10-0.12)
- Combined: eta_eff = eta_T * eta_pl

## References
- Preuster et al. (2017). Acc. Chem. Res. 50(1), 74-85.
- Niermann et al. (2021). Energy Environ. Sci. 14, 1928-1944.
- Reuse et al. (2004). Chem. Eng. J. 101(1-3), 133-141.
- Lamb et al. (2019). Int. J. Hydrogen Energy 44(7), 3580-3593.
