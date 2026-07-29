"""
EC096 — Magnetic Refrigeration — F1b COP vs Temperature + Part-Load Penalty

Extends F1a (basic COP map) with:
  1. Thermodynamic Carnot-based COP corrected for magnetocaloric material performance
  2. Temperature-dependent COP: COP = f(T_hot, T_cold, delta_T_span)
  3. Part-load penalty curve (magnetic refrigerators do not throttle well)
  4. Active Magnetic Regenerator (AMR) efficiency factor

Physics:
    Carnot COP = T_cold / (T_hot - T_cold)
    AMR COP    = COP_Carnot * eta_AMR * f_PLR(PLR) * f_T(T_hot)

Where:
    eta_AMR = eta_magnet * eta_regen * eta_cycle
         — accounts for magnet work inefficiency, regenerator losses, motor losses

    f_T(T_hot) = correction for hot-side temperature deviation:
         f_T = 1 - k_T * (T_hot - T_hot_design)
         (Higher reject temp requires more magnetic work)

    f_PLR(PLR) = part-load penalty:
         At part load, magnetic refrigerators are less efficient because
         the magnetocaloric effect (MCE) is optimised at rated frequency.
         f_PLR = p1 + p2*PLR + p3*PLR^2  (quadratic, peak near PLR=1.0)

Cooling capacity:
    Q_cold = COP_Carnot * eta_AMR * f_PLR * f_T * W_in   [kW]
or equivalently:
    COP_actual = COP_Carnot * eta_AMR * f_PLR * f_T

References:
    Kitanovski et al. (2015), Magnetocaloric Energy Conversion, Springer.
    Aprea et al. (2015), 'A first and second law analysis of a transcritical CO2 and
    magnetic refrigeration system', Int. J. Refrig. 52, 98-108.
    Yu et al. (2010), 'The development of the AMR refrigeration cycle', Int. J. Refrig.
    33(6), 1029-1060.
"""

import numpy as np


class MagneticRefrigerationF1b:
    """Magnetic refrigeration — thermodynamic COP with part-load and T correction."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.Q_rated = u["Q_rated_kw"]["value"]         # kW cooling capacity
        self.T_hot_design = u["T_hot_design_c"]["value"] # degC
        self.T_cold_design = u["T_cold_design_c"]["value"] # degC
        self.eta_AMR = u["eta_AMR"]["value"]             # AMR system efficiency vs Carnot
        self.k_T = u["k_T"]["value"]                     # 1/K — hot-temp sensitivity
        self.PLR_min = u["PLR_min"]["value"]
        self.p1 = u["plr_p1"]["value"]
        self.p2 = u["plr_p2"]["value"]
        self.p3 = u["plr_p3"]["value"]

    # ------------------------------------------------------------------
    # Carnot COP
    # ------------------------------------------------------------------

    def cop_carnot(self, T_hot_c, T_cold_c):
        """Carnot COP = T_cold_K / (T_hot_K - T_cold_K)."""
        T_hot  = np.asarray(T_hot_c, dtype=float) + 273.15
        T_cold = np.asarray(T_cold_c, dtype=float) + 273.15
        dT = T_hot - T_cold
        return np.where(dT > 0.1, T_cold / dT, 100.0)  # avoid division by zero

    # ------------------------------------------------------------------
    # Temperature correction
    # ------------------------------------------------------------------

    def f_temp(self, T_hot_c):
        """
        COP correction for hot-side temperature deviation from design.
        f_T = 1 - k_T * (T_hot - T_hot_design)
        Lower f_T when reject temperature rises.
        """
        T_hot = np.asarray(T_hot_c, dtype=float)
        f = 1.0 - self.k_T * (T_hot - self.T_hot_design)
        return np.clip(f, 0.2, 1.3)

    # ------------------------------------------------------------------
    # Part-load correction
    # ------------------------------------------------------------------

    def f_plr(self, PLR):
        """
        Part-load COP correction (quadratic).
        At PLR=1: f_PLR = p1+p2+p3 = 1.0 (by design).
        At PLR < 1: reduced because AMR frequency not optimal.
        """
        PLR = np.asarray(PLR, dtype=float)
        PLR_eff = np.maximum(PLR, self.PLR_min)
        return self.p1 + self.p2 * PLR_eff + self.p3 * PLR_eff**2

    # ------------------------------------------------------------------
    # Actual COP
    # ------------------------------------------------------------------

    def cop(self, T_hot_c, T_cold_c, PLR=1.0):
        """
        Actual system COP.
        COP = COP_Carnot * eta_AMR * f_PLR * f_T
        """
        cop_c = self.cop_carnot(T_hot_c, T_cold_c)
        f_t   = self.f_temp(T_hot_c)
        f_p   = self.f_plr(PLR)
        cop_val = cop_c * self.eta_AMR * f_t * f_p
        # Magnetic refrigerators: COP > 1.0 (cooling machine, not heat engine)
        # Upper bound based on realistic AMR performance
        return np.clip(cop_val, 0.5, 8.0)

    # ------------------------------------------------------------------
    # Cooling capacity and power
    # ------------------------------------------------------------------

    def cooling_capacity_kw(self, PLR=1.0):
        """Cooling output [kW]."""
        PLR = np.maximum(np.asarray(PLR, dtype=float), self.PLR_min)
        return self.Q_rated * PLR

    def electrical_input_kw(self, T_hot_c, T_cold_c, PLR=1.0):
        """Electrical (or magnet-drive) power input [kW]."""
        Q_c = self.cooling_capacity_kw(PLR)
        cop_val = self.cop(T_hot_c, T_cold_c, PLR)
        return Q_c / cop_val

    def heat_rejection_kw(self, T_hot_c, T_cold_c, PLR=1.0):
        """Heat rejected to hot sink [kW]."""
        Q_c = self.cooling_capacity_kw(PLR)
        W   = self.electrical_input_kw(T_hot_c, T_cold_c, PLR)
        return Q_c + W

    # ------------------------------------------------------------------
    # Specific work
    # ------------------------------------------------------------------

    def delta_T_span(self, T_hot_c, T_cold_c):
        """Temperature span across AMR [K]."""
        return np.asarray(T_hot_c, dtype=float) - np.asarray(T_cold_c, dtype=float)

    # ------------------------------------------------------------------
    # predict_all
    # ------------------------------------------------------------------

    def predict_all(self, T_hot_c, T_cold_c, PLR=1.0):
        """
        Return all outputs as a dict.

        Parameters
        ----------
        T_hot_c  : hot-reservoir (reject) temperature [degC]
        T_cold_c : cold-reservoir (load) temperature [degC]
        PLR      : part-load ratio [0.3-1.0]
        """
        cop_c_val  = self.cop_carnot(T_hot_c, T_cold_c)
        cop_val    = self.cop(T_hot_c, T_cold_c, PLR)
        Q_c        = self.cooling_capacity_kw(PLR)
        W_in       = self.electrical_input_kw(T_hot_c, T_cold_c, PLR)
        Q_hot      = self.heat_rejection_kw(T_hot_c, T_cold_c, PLR)
        dT_span    = self.delta_T_span(T_hot_c, T_cold_c)
        eta_ratio  = cop_val / np.maximum(cop_c_val, 1e-6)  # eta/Carnot

        return {
            "cop":               cop_val,
            "cop_carnot":        cop_c_val,
            "eta_vs_carnot":     np.clip(eta_ratio, 0.0, 1.0),
            "cooling_kw":        Q_c,
            "electrical_kw":     W_in,
            "heat_rejection_kw": Q_hot,
            "delta_T_span_K":    dT_span,
        }
