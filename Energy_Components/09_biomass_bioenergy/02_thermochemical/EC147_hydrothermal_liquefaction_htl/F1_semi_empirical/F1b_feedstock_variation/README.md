# EC147 Hydrothermal Liquefaction (HTL) — F1b Feedstock Variation

## Model Summary
Feedstock-specific HTL model with biochemical composition driving bio-crude yield, moisture-LHV coupling, Arrhenius temperature factor peaking at ~330°C, and product phase distribution.

## Key Physics
- **Moisture-LHV coupling**: `LHV_eff = LHV_dry*(1-M) - h_fg*M` (M up to 90%)
- **Bio-crude yield**: `Y_bc = (0.45*lipid + 0.25*protein + 0.12*carb) * f_T * f_moisture`
- **Temperature factor**: Arrhenius peak at ~330°C; secondary cracking above 350°C
- **Moisture benefit**: Subcritical water medium improves extraction for wet feedstocks

## References
- Peterson, A.A. et al. (2008). Energy & Environ. Sci., 1, 32-65.
- Anastasakis, K. & Ross, A.B. (2011). Bioresource Technology, 102, 4876-4883.
- Vardon, D.R. et al. (2011). J. Anal. Appl. Pyrolysis, 91, 108-117.
- Elliott, D.C. et al. (2015). Biofuels, Bioprod. Bioref., 9, 507-527.
