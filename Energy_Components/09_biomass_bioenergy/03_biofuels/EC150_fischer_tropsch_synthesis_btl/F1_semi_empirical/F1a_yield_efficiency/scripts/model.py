"""
EC150 -- Fischer-Tropsch Synthesis (BTL) -- F1a Yield Model

Simplest semi-empirical model using fixed CO conversion and ASF distribution:
    CO_reacted   = syngas_CO_mol_per_h * CO_conversion
    FT_liquid_kg = CO_reacted * ASF_liquid_fraction * (MW_CH2/MW_CO) [simplified]

ASF (Anderson-Schulz-Flory) chain-growth distribution:
    W_n = n * (1-alpha)^2 * alpha^(n-1)   [weight fraction for carbon number n]
    Liquid fraction (C5+) = sum(W_n for n>=5) ~ depends on alpha

For alpha=0.85: liquid fraction ~0.78 of total HC product.

Simplified mass yield approach (avoids full ASF summation):
    FT_products_kg_per_Nm3_CO = 0.625 * CO_conversion   [empirical, CO basis]

References:
    Dry, M.E. (2002). Catal. Today 71:227.
    Steynberg, A. & Dry, M. (2004). Fischer-Tropsch Technology. Elsevier.
    Spath, P.L. & Dayton, D.C. (2003). NREL/TP-510-34929.
"""

import numpy as np


class FischerTropschF1a:
    """FT synthesis: liquid_products = syngas_CO_flow * CO_conversion * FT_yield."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.CO_conversion    = float(u["CO_conversion"]["value"])
        self.alpha_ASF        = float(u["alpha_ASF"]["value"])
        self.T_operating      = float(u["T_operating_degC"]["value"])
        self.P_operating      = float(u["P_operating_bar"]["value"])
        self.H2_CO_ratio      = float(u["H2_CO_ratio"]["value"])
        self.LHV_FT           = float(u["LHV_FT_liquid_MJ_kg"]["value"])
        # Product distribution fractions
        self.wax_frac         = float(u["wax_fraction_C21plus"]["value"])
        self.diesel_frac      = float(u["diesel_fraction_C10_C20"]["value"])
        self.naphtha_frac     = float(u["naphtha_fraction_C5_C9"]["value"])
        self.light_gas_frac   = float(u["light_gas_fraction_C1_C4"]["value"])

        # Simplified empirical yield: kg FT liquid per Nm3 CO fed (Dry 2002)
        # At CO_conversion 0.80: ~0.625 kg liquid per Nm3 CO (CO molar mass basis)
        # FT stoichiometry: CO + 2H2 -> -CH2- + H2O; MW_CH2=14, MW_CO=28
        # liquid_frac of total HC (C5+) from ASF at alpha=0.85 ~ 0.78
        liquid_frac_ASF = self._asf_liquid_fraction(self.alpha_ASF)
        # kg HC per Nm3 CO: (1/22.4) mol_CO * 28 g/mol -> 1.25 g CO/L -> per Nm3: 28/22.4 = 1.25 kg/Nm3
        # Mass of HC = mass_CO * (14/28) = 0.5 * mass_CO (per C atom)
        # Total HC per Nm3 CO fed (at full conversion): 28/22.4 * 0.5 = 0.625 kg/Nm3
        self.yield_kg_per_Nm3_CO_converted = 0.625 * liquid_frac_ASF

    @staticmethod
    def _asf_liquid_fraction(alpha: float) -> float:
        """Liquid fraction (C5+) from ASF distribution."""
        # W_n = n*(1-alpha)^2*alpha^(n-1); liquid = sum(n=5..inf)
        # Analytical: C5+ fraction = 1 - sum(n=1..4) W_n
        frac_light = 0.0
        for n in range(1, 5):
            frac_light += n * (1 - alpha)**2 * alpha**(n - 1)
        return max(0.0, 1.0 - frac_light)

    def predict(self, syngas_flow_Nm3_per_h: float, CO_fraction_in: float = 0.40) -> dict:
        """
        Parameters
        ----------
        syngas_flow_Nm3_per_h : float  Total syngas flow [Nm3/h]
        CO_fraction_in        : float  CO mole fraction in syngas [-]

        Returns
        -------
        dict with:
            CO_reacted_Nm3_per_h    : CO consumed [Nm3/h]
            FT_liquid_kg_per_h      : total liquid product [kg/h]
            diesel_kg_per_h         : diesel-range product [kg/h]
            naphtha_kg_per_h        : naphtha product [kg/h]
            wax_kg_per_h            : wax product [kg/h]
            light_gas_kg_per_h      : light gases [kg/h]
            energy_output_MW        : energy in FT liquid [MW]
            CO_conversion           : CO conversion fraction [-]
        """
        Q_syngas  = float(np.clip(syngas_flow_Nm3_per_h, 0.0, None))
        CO_frac   = float(np.clip(CO_fraction_in, 0.0, 1.0))

        CO_fed     = Q_syngas * CO_frac                             # Nm3/h CO fed
        CO_reacted = CO_fed * self.CO_conversion                    # Nm3/h CO reacted
        FT_total   = CO_reacted * self.yield_kg_per_Nm3_CO_converted # kg/h all HC products

        # Distribute total HC by ASF fractions (re-normalized to HC products only)
        diesel   = FT_total * self.diesel_frac
        naphtha  = FT_total * self.naphtha_frac
        wax      = FT_total * self.wax_frac
        light    = FT_total * self.light_gas_frac

        # Energy: count liquid (C5+) only
        FT_liquid   = diesel + naphtha + wax
        energy_out  = FT_liquid * self.LHV_FT / 3600.0   # MW

        return {
            "CO_reacted_Nm3_per_h": float(CO_reacted),
            "FT_liquid_kg_per_h":   float(FT_liquid),
            "diesel_kg_per_h":      float(diesel),
            "naphtha_kg_per_h":     float(naphtha),
            "wax_kg_per_h":         float(wax),
            "light_gas_kg_per_h":   float(light),
            "energy_output_MW":     float(energy_out),
            "CO_conversion":        float(self.CO_conversion),
        }
