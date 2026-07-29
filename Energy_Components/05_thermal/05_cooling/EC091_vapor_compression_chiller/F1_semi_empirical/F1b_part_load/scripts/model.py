"""
EC091 — Vapor Compression Chiller — F1b Part-Load Model

Extends F1a with IPLV (Integrated Part Load Value) methodology:

  COP(PLR, T_cw) = COP_ref / EIR_fPLR(PLR) * f_T(T_cw)

  EIR_fPLR(PLR) = d1 + d2*PLR + d3*PLR^2      [DOE-2 chiller curve]
  f_T(T_cw) = e1 + e2*T_cw + e3*T_cw^2        [condenser water correction]

  IPLV = 1 / (0.01/COP_100 + 0.42/COP_75 + 0.45/COP_50 + 0.12/COP_25)

At part load, the chiller operates at lower condenser water temperatures
(AHRI unloading), which improves COP. The IPLV captures this benefit.

References:
    AHRI Standard 550/590 — Performance rating of water-chilling packages.
    DOE-2 Reference Manual, Chiller curves.
    EnergyPlus Engineering Reference (2023), Chiller:Electric:EIR.
    Gordon & Ng (2000). Cool Thermodynamics.
"""

import numpy as np


class ChillerF1b:
    """Vapor compression chiller with IPLV part-load methodology."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.Q_rated = u["Q_rated"]["value"]
        self.COP_rated = u["COP_rated"]["value"]
        self.T_chw_rated = u["T_chw_rated"]["value"]
        self.T_cw_rated = u["T_cw_rated"]["value"]
        self.PLR_min = u["PLR_min"]["value"]

        # DOE-2 part-load curve
        plr_c = u["plr_curve"]
        self.d1 = plr_c["d1"]["value"]
        self.d2 = plr_c["d2"]["value"]
        self.d3 = plr_c["d3"]["value"]

        # Temperature correction curve
        tc = u["temp_curve"]
        self.e1 = tc["e1"]["value"]
        self.e2 = tc["e2"]["value"]
        self.e3 = tc["e3"]["value"]

        # IPLV weights and temperatures
        iw = u["iplv_weights"]
        self.iplv_w = [iw["w100"], iw["w75"], iw["w50"], iw["w25"]]
        it = u["iplv_temps"]
        self.iplv_T = [
            it["T_cw_100"]["value"], it["T_cw_75"]["value"],
            it["T_cw_50"]["value"], it["T_cw_25"]["value"],
        ]
        self.iplv_plr = [1.0, 0.75, 0.50, 0.25]

    # ------------------------------------------------------------------
    # Part-load EIR curve
    # ------------------------------------------------------------------

    def eir_f_plr(self, plr):
        """
        EIR/EIR_ref as a function of PLR (DOE-2 curve).
        At PLR=1: should be ~1.0 (d1+d2+d3=1.0).
        At low PLR: EIR ratio decreases (chiller more efficient per unit cooling).
        """
        plr = np.asarray(plr, dtype=float)
        plr_eff = np.maximum(plr, self.PLR_min)
        eir = self.d1 + self.d2 * plr_eff + self.d3 * plr_eff ** 2
        return np.clip(eir, 0.1, 2.0)

    # ------------------------------------------------------------------
    # Temperature correction
    # ------------------------------------------------------------------

    def f_temp(self, T_cw):
        """
        COP correction factor for condenser water temperature.
        f_T > 1 when T_cw < T_cw_rated (lower condensing temp = better COP).
        f_T < 1 when T_cw > T_cw_rated.
        """
        T_cw = np.asarray(T_cw, dtype=float)
        return self.e1 + self.e2 * T_cw + self.e3 * T_cw ** 2

    # ------------------------------------------------------------------
    # COP calculation
    # ------------------------------------------------------------------

    def cop(self, T_chw, T_cw, plr=1.0):
        """
        COP at given operating conditions.

        COP = COP_rated * f_T(T_cw) / EIR_fPLR(PLR)

        Note: dividing by EIR because EIR is the inverse of COP multiplier.
        At PLR=1, T_cw=rated: COP = COP_rated.
        """
        plr = np.asarray(plr, dtype=float)
        plr_eff = np.maximum(plr, self.PLR_min)

        eir_ratio = self.eir_f_plr(plr_eff)
        f_t = self.f_temp(T_cw)

        # COP = COP_rated * f_T / eir_ratio
        # But eir_ratio at PLR=1 = d1+d2+d3 = 1.0, so COP = COP_rated * f_T at rated
        cop_val = self.COP_rated * f_t / eir_ratio
        return np.clip(cop_val, 1.0, 25.0)

    # ------------------------------------------------------------------
    # IPLV calculation
    # ------------------------------------------------------------------

    def iplv(self):
        """
        Integrated Part Load Value per AHRI 550/590.

        IPLV = 1 / sum(w_i / COP_i)

        where COP_i is at the corresponding PLR and T_cw.
        """
        cop_values = []
        for plr_i, T_cw_i in zip(self.iplv_plr, self.iplv_T):
            c = float(self.cop(self.T_chw_rated, T_cw_i, plr_i))
            cop_values.append(c)

        weighted_inv = sum(w / c for w, c in zip(self.iplv_w, cop_values))
        return 1.0 / weighted_inv

    # ------------------------------------------------------------------
    # Capacity and power
    # ------------------------------------------------------------------

    def cooling_capacity(self, plr=1.0):
        """Cooling output in kW."""
        return self.Q_rated * np.asarray(plr, dtype=float)

    def electrical_input(self, T_chw, T_cw, plr=1.0):
        """Compressor electrical input in kW."""
        q = self.cooling_capacity(plr)
        c = self.cop(T_chw, T_cw, plr)
        return q / c
