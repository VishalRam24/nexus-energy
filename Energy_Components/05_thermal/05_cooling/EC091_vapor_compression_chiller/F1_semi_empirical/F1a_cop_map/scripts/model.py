"""
EC091 — Vapor Compression Chiller — F1a COP Map Model

COP = eta_Carnot * COP_Carnot = eta_Carnot * T_evap / (T_cond - T_evap)
Part-load: COP(PLR) = COP_full * (c1 + c2*PLR + c3*PLR^2)
Q_cooling = Q_rated * PLR
W_comp = Q_cooling / COP

Reference:
    Gordon & Ng (2000). Cool Thermodynamics.
    Cambridge International Science Publishing.
"""

import numpy as np


class ChillerF1a:
    """Vapor compression chiller — COP as a function of evap/cond temperatures + PLR."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.Q_rated = u["Q_rated"]["value"]          # kW cooling
        self.carnot_fraction = u["carnot_fraction"]["value"]
        self.COP_rated = u["COP_rated"]["value"]
        plr_c = u["plr_coefficients"]
        self.c1 = plr_c["c1"]["value"]
        self.c2 = plr_c["c2"]["value"]
        self.c3 = plr_c["c3"]["value"]

    def cop_full_load(self, T_evap_c, T_cond_c):
        """Full-load COP from Carnot fraction approach.

        Args:
            T_evap_c: Evaporator (chilled water supply) temperature in degC.
            T_cond_c: Condenser temperature in degC.

        Returns:
            COP (dimensionless, clipped to [1.0, 20.0]).
        """
        T_evap = np.asarray(T_evap_c, dtype=float) + 273.15
        T_cond = np.asarray(T_cond_c, dtype=float) + 273.15
        dT = T_cond - T_evap
        cop_carnot = np.where(dT > 0, T_evap / dT, 20.0)
        cop = self.carnot_fraction * cop_carnot
        return np.clip(cop, 1.0, 20.0)

    def plr_factor(self, plr):
        """Part-load ratio correction factor (Gordon-Ng polynomial)."""
        plr = np.asarray(plr, dtype=float)
        return self.c1 + self.c2 * plr + self.c3 * plr ** 2

    def cop(self, T_evap_c, T_cond_c, plr=1.0):
        """COP at given temperatures and part-load ratio."""
        cop_fl = self.cop_full_load(T_evap_c, T_cond_c)
        f = self.plr_factor(plr)
        return cop_fl * f

    def cooling_power(self, plr=1.0):
        """Cooling output in kW."""
        return self.Q_rated * np.asarray(plr, dtype=float)

    def compressor_power(self, T_evap_c, T_cond_c, plr=1.0):
        """Compressor electrical input in kW."""
        q = self.cooling_power(plr)
        c = self.cop(T_evap_c, T_cond_c, plr)
        return q / c

    def heat_rejection(self, T_evap_c, T_cond_c, plr=1.0):
        """Heat rejected to condenser in kW (Q_cool + W_comp)."""
        q = self.cooling_power(plr)
        w = self.compressor_power(T_evap_c, T_cond_c, plr)
        return q + w
