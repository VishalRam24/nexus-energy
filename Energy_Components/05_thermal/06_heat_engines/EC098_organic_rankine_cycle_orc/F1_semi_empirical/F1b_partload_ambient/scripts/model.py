"""
EC098 -- Organic Rankine Cycle (ORC) -- F1b Part-Load + Condenser Ambient Effect

Extends F1a by adding:
  1. Quadratic part-load correction (better off-design pump/expander match)
  2. Condenser temperature effect on efficiency (very sensitive for ORC)
  3. Specific work output

Efficiency model:
    eta_carnot = 1 - T_cold_K / T_hot_K
    eta_design = eta_carnot * eta_internal

    Part-load correction:
        f_PLR(PLR) = a + b*PLR + c*PLR^2
        (Peaks near PLR~0.85-1.0; at 50% load, f ~ 0.85)

    Condenser temperature correction:
        f_T = 1 - k_T * (T_cond - T_cond_design)
        ORC efficiency is very sensitive to condenser temperature because
        the Carnot efficiency is low (~20-30%), so a small absolute change
        in T_cold causes a large relative efficiency change.

    Combined:
        eta = eta_design * f_PLR * f_T

Power output:
    P_out = Q_hot * eta

Heat rejection:
    Q_reject = Q_hot - P_out

Specific work:
    w_specific = eta * (h_hot - h_cold) [approximated from temperatures]

References:
    Quoilin et al. (2013), Techno-economic survey of ORC systems.
    Manente et al. (2013), Off-design performance of ORC power plants.
    Lecompte et al. (2015), Review of ORC for low-grade waste heat recovery.
"""

import numpy as np


class ORCF1b:
    """ORC with part-load and condenser ambient temperature correction."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.P_rated       = u["P_rated_kw"]["value"]          # kW_e
        self.T_hot_design  = u["T_hot_design_c"]["value"]      # degC
        self.T_cond_design = u["T_cond_design_c"]["value"]     # degC
        self.eta_internal  = u["eta_internal"]["value"]
        self.PLR_min       = u["PLR_min"]["value"]
        self.plr_a         = u["plr_a"]["value"]
        self.plr_b         = u["plr_b"]["value"]
        self.plr_c         = u["plr_c"]["value"]
        self.k_T           = u["T_cond_sensitivity"]["value"]  # 1/K
        self.T_approach    = u["T_approach_c"]["value"]        # degC

    # ------------------------------------------------------------------
    # Correction factors
    # ------------------------------------------------------------------

    def eta_carnot(self, T_hot_c, T_cond_c):
        """Carnot efficiency (upper bound)."""
        T_hot = np.asarray(T_hot_c, dtype=float) + 273.15
        T_cold = np.asarray(T_cond_c, dtype=float) + 273.15
        dT = T_hot - T_cold
        return np.where(dT > 0.0, 1.0 - T_cold / T_hot, 0.0)

    def f_plr(self, PLR):
        """Part-load correction (quadratic)."""
        PLR = np.asarray(PLR, dtype=float)
        return self.plr_a + self.plr_b * PLR + self.plr_c * PLR ** 2

    def f_condenser(self, T_cond_c):
        """Condenser temperature correction.
        f = 1 - k_T * (T_cond - T_cond_design)
        Higher condenser temp -> lower efficiency (very sensitive for ORC).
        """
        T = np.asarray(T_cond_c, dtype=float)
        f = 1.0 - self.k_T * (T - self.T_cond_design)
        return np.clip(f, 0.3, 1.5)  # allow boost from cold condenser, but cap

    # ------------------------------------------------------------------
    # Efficiency
    # ------------------------------------------------------------------

    def efficiency(self, T_hot_c, T_cond_c, PLR=1.0):
        """Net ORC thermal efficiency."""
        eta_c = self.eta_carnot(T_hot_c, T_cond_c)
        eta_design = eta_c * self.eta_internal
        f_p = self.f_plr(PLR)
        f_t = self.f_condenser(T_cond_c)
        eta = eta_design * f_p * f_t
        # Must not exceed Carnot
        return np.clip(eta, 0.0, eta_c)

    # ------------------------------------------------------------------
    # Power and heat flows
    # ------------------------------------------------------------------

    def power_output_kw(self, T_hot_c, T_cond_c, PLR=1.0, heat_input_kw=None):
        """Electrical power output [kW]."""
        eta = self.efficiency(T_hot_c, T_cond_c, PLR)
        Q_hot = self._heat_input(T_hot_c, T_cond_c, PLR, heat_input_kw)
        return Q_hot * eta

    def heat_rejection_kw(self, T_hot_c, T_cond_c, PLR=1.0, heat_input_kw=None):
        """Heat rejection to condenser [kW]."""
        Q_hot = self._heat_input(T_hot_c, T_cond_c, PLR, heat_input_kw)
        P_out = self.power_output_kw(T_hot_c, T_cond_c, PLR, heat_input_kw)
        return Q_hot - P_out

    def specific_work_kj_kg(self, T_hot_c, T_cond_c, PLR=1.0):
        """Approximate specific work [kJ/kg].
        Estimated from efficiency and enthalpy difference approximation.
        For R245fa: cp ~ 1.3 kJ/(kg.K), assuming superheat of ~(T_hot - T_boil).
        """
        eta = self.efficiency(T_hot_c, T_cond_c, PLR)
        T_hot = np.asarray(T_hot_c, dtype=float)
        T_cond = np.asarray(T_cond_c, dtype=float)
        # Approximate enthalpy drop available: cp * (T_hot - T_cond)
        cp_approx = 1.3  # kJ/(kg.K) for R245fa
        dh_available = cp_approx * (T_hot - T_cond)
        return eta * dh_available

    def _heat_input(self, T_hot_c, T_cond_c, PLR, heat_input_kw):
        """Resolve heat input: user-specified or back-calculated from rated."""
        PLR_arr = np.asarray(PLR, dtype=float)
        if heat_input_kw is not None:
            return np.asarray(heat_input_kw, dtype=float)
        # Back-calculate from rated conditions
        eta_rated = float(self.efficiency(self.T_hot_design, self.T_cond_design, 1.0))
        Q_hot_rated = self.P_rated / max(eta_rated, 1e-6)
        return Q_hot_rated * PLR_arr
