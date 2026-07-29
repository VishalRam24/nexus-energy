# EC146 Torrefaction Reactor — F1b Feedstock Variation

## Model Summary
Feedstock-specific torrefaction model adding moisture-LHV coupling, temperature/residence-time dependent mass yield, and energy densification ratio to F1a.

## Key Physics
- **Moisture-LHV coupling**: `LHV_eff = LHV_dry*(1-M) - h_fg*M`
- **Mass yield**: `MY = exp(-k_m * dT^n * t^p)`, hemicellulose-rich feedstocks degrade faster
- **Energy densification**: `EDR = 1 + a_edr*(T-200)*(t/30)^0.3 * lignin_factor`, capped at 1.35
- **Part-load**: `eta = (a0 + a1*PLR + a2*PLR²) * moisture_factor`

## References
- Bach, Q.V. et al. (2017). Fuel, 202, 573-578.
- van der Stelt, M.J.C. et al. (2011). Biomass & Bioenergy, 35(9), 3748-3762.
- Bergman, P.C.A. et al. (2005). ECN-C-05-013.
