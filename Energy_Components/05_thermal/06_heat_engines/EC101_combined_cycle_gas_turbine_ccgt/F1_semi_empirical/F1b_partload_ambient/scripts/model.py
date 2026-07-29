"""
EC101 -- Combined Cycle Gas Turbine (CCGT) -- F1b Part-Load + Ambient

Extends F1a by modelling GT and ST separately:

GT part-load:
    eta_GT(PLR) = eta_GT_rated * (a + b*PLR + c*PLR^2)

GT exhaust temperature:
    T_exhaust_GT ~ T_rated + dT*(1 - PLR)
    (At lower GT load, TIT drops but mass flow drops faster, so exhaust
     temperature can actually *increase* slightly -- maintains HRSG perf.)

ST (bottoming cycle):
    Receives GT exhaust heat: Q_exhaust = P_GT * (1/eta_GT - 1)
    eta_ST(PLR) = eta_ST_rated * (st_a + st_b * PLR_ST)
    where PLR_ST is the ST loading fraction based on available exhaust heat
    vs design exhaust heat.

Combined efficiency:
    eta_combined = (P_GT + P_ST) / Q_fuel

Ambient corrections (applied to GT):
    P_GT_corrected = P_GT_iso * (P_amb/P_ref) * sqrt(T_ref/T_amb)
    eta_GT also adjusted by sqrt(T_ref/T_amb)

References:
    Kehlhofer, R. et al. (2009). Combined-Cycle Gas & Steam Turbine Power Plants.
    Chase, D.L. (2001). Combined-Cycle Development, Evolution, and Future.
    GE Power H-class reference data.
"""

import numpy as np


class CCGTF1b:
    """CCGT with separate GT/ST part-load models and ambient correction."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.P_rated       = u["rated_power_mw"]["value"]          # MW combined
        self.eta_gt_rated  = u["eta_gt_rated"]["value"]
        self.eta_st_rated  = u["eta_st_rated"]["value"]
        self.eta_comb_rated = u["eta_combined_rated"]["value"]
        self.T_iso_ref     = u["T_iso_ref_k"]["value"]             # K
        self.P_iso_ref     = u["P_iso_ref_kpa"]["value"]           # kPa
        self.gt_a          = u["gt_plr_a"]["value"]
        self.gt_b          = u["gt_plr_b"]["value"]
        self.gt_c          = u["gt_plr_c"]["value"]
        self.st_a          = u["st_plr_a"]["value"]
        self.st_b          = u["st_plr_b"]["value"]
        self.LHV           = u["LHV_gas_mj_kg"]["value"]           # MJ/kg
        self.PLR_min       = u["PLR_min"]["value"]
        self.T_exh_gt_rated = u["T_exhaust_gt_rated_k"]["value"]   # K
        self.dT_partload   = u["T_exhaust_gt_partload_rise_k"]["value"]
        self.T_stack       = u["T_stack_k"]["value"]               # K

        # Derive GT fraction of total rated power
        # At rated: P_GT = fuel * eta_GT, P_ST = fuel * (1 - eta_GT) * eta_ST_eff
        # eta_combined = eta_GT + (1 - eta_GT) * eta_ST_eff
        # Solve for eta_ST_eff: eta_ST_eff = (eta_comb - eta_GT) / (1 - eta_GT)
        self.eta_st_eff_rated = (self.eta_comb_rated - self.eta_gt_rated) / (1.0 - self.eta_gt_rated)
        # GT share of power: P_GT/P_total = eta_GT / eta_combined
        self.gt_power_frac = self.eta_gt_rated / self.eta_comb_rated

    # ------------------------------------------------------------------
    # GT model
    # ------------------------------------------------------------------

    def f_plr_gt(self, PLR):
        """GT part-load efficiency correction factor."""
        PLR = np.asarray(PLR, dtype=float)
        return self.gt_a + self.gt_b * PLR + self.gt_c * PLR ** 2

    def f_amb_power(self, T_amb_k, P_amb_kpa):
        """ISO power correction for GT."""
        T = np.asarray(T_amb_k, dtype=float)
        P = np.asarray(P_amb_kpa, dtype=float)
        return (P / self.P_iso_ref) * np.sqrt(self.T_iso_ref / T)

    def f_amb_eta(self, T_amb_k):
        """Efficiency correction for GT due to ambient temperature."""
        T = np.asarray(T_amb_k, dtype=float)
        return np.sqrt(self.T_iso_ref / T)

    def eta_gt(self, PLR, T_amb_k):
        """GT efficiency."""
        eta = self.eta_gt_rated * self.f_plr_gt(PLR) * self.f_amb_eta(T_amb_k)
        return np.clip(eta, 1e-6, 0.45)

    def exhaust_temp_gt_k(self, PLR):
        """GT exhaust temperature [K] -- increases slightly at part load."""
        PLR = np.asarray(PLR, dtype=float)
        return self.T_exh_gt_rated + self.dT_partload * (1.0 - PLR)

    # ------------------------------------------------------------------
    # ST (bottoming cycle) model
    # ------------------------------------------------------------------

    def eta_st(self, PLR):
        """ST efficiency as fraction of available exhaust heat.
        At part load, HRSG sees different exhaust conditions.
        PLR_ST approximated as proportional to GT exhaust heat fraction.
        """
        PLR = np.asarray(PLR, dtype=float)
        f_st = self.st_a + self.st_b * PLR
        eta = self.eta_st_eff_rated * f_st
        return np.clip(eta, 0.0, 0.45)

    # ------------------------------------------------------------------
    # Combined outputs
    # ------------------------------------------------------------------

    def efficiency_combined(self, PLR, T_amb_k, P_amb_kpa=101.325):
        """Combined cycle net efficiency.
        eta_cc = eta_GT + (1 - eta_GT) * eta_ST_eff
        """
        PLR = np.asarray(PLR, dtype=float)
        e_gt = self.eta_gt(PLR, T_amb_k)
        e_st = self.eta_st(PLR)
        eta_cc = e_gt + (1.0 - e_gt) * e_st
        return np.clip(eta_cc, 1e-6, 0.70)

    def efficiency_gt_out(self, PLR, T_amb_k, P_amb_kpa=101.325):
        """GT efficiency (for output reporting)."""
        return self.eta_gt(PLR, T_amb_k)

    def efficiency_st_out(self, PLR, T_amb_k, P_amb_kpa=101.325):
        """ST effective efficiency (for output reporting)."""
        return self.eta_st(PLR)

    def power_output_kw(self, PLR, T_amb_k, P_amb_kpa=101.325):
        """Combined electrical output [kW].
        P = P_rated * PLR * f_amb_power (ambient affects GT, ST follows)
        """
        PLR = np.asarray(PLR, dtype=float)
        f_amb = self.f_amb_power(T_amb_k, P_amb_kpa)
        return self.P_rated * PLR * f_amb * 1e3  # MW -> kW

    def heat_rate_kj_kwh(self, PLR, T_amb_k, P_amb_kpa=101.325):
        """Combined heat rate [kJ/kWh]."""
        eta = self.efficiency_combined(PLR, T_amb_k, P_amb_kpa)
        return 3600.0 / eta

    def exhaust_temp_k(self, PLR):
        """Stack exhaust temperature [K] (after HRSG)."""
        # HRSG stack temp is roughly constant (~80C) but rises at low load
        PLR = np.asarray(PLR, dtype=float)
        return self.T_stack + 20.0 * (1.0 - PLR)
