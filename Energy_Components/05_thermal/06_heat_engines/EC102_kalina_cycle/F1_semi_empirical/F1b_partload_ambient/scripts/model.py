"""
EC102 — Kalina Cycle — F1b Part-Load + Condenser Temperature Model

The Kalina cycle uses an ammonia-water (NH3-H2O) mixture as the working fluid.
The variable composition boiling/condensing provides better thermal matching
to heat sources and sinks than pure-fluid ORC.

Extends F1a (basic thermal efficiency) with:
  1. Ammonia mass fraction effect on cycle efficiency (key advantage of Kalina)
  2. Condenser temperature sensitivity (ammonia condenser pressure highly T-sensitive)
  3. Part-load efficiency correction
  4. Heat source temperature correction via modified Carnot approach

Kalina cycle efficiency model:
    eta_Carnot = 1 - T_cond_K / T_hot_K
    eta_design = eta_Carnot * eta_internal
    f_x = 1 + k_x * (x_NH3 - x_NH3_design)    [ammonia fraction correction]
    f_T = 1 - k_T * (T_cond - T_cond_design)    [condenser T sensitivity]
    f_PLR = a + b*PLR + c*PLR^2

    eta = eta_design * f_x * f_T * f_PLR

Advantage over ORC:
    - Ammonia content can be tuned: for waste heat at 100-200C,
      optimal x_NH3 ~ 0.8-0.9 gives 10-20% better efficiency than ORC.
    - f_x coefficient: k_x > 0 means higher NH3 fraction improves efficiency
      up to an optimum, then decreases (model is valid around ±10% of design).

References:
    Kalina (1984), US Patent 4,346,561. Combined cycle system with novel bottoming cycle.
    Leibowitz et al. (1997), 'Kalina cycle looks good for geothermal applications',
    Modern Power Systems, June 1997.
    Bombarda et al. (2010), 'Heat recovery from diesel engines: A thermodynamic
    comparison between Kalina and ORC cycles', Appl. Thermal Eng. 30(2), 212-219.
    Lolos & Rogdakis (2009), 'A Kalina power cycle driven by renewable energy sources',
    Energy 34(4), 457-464.
"""

import numpy as np


class KalinaCycleF1b:
    """Kalina cycle — part-load + condenser T + ammonia fraction model."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.P_rated        = u["P_rated_kw"]["value"]         # kW_e
        self.T_hot_design   = u["T_hot_design_c"]["value"]     # degC
        self.T_cond_design  = u["T_cond_design_c"]["value"]    # degC
        self.x_NH3_design   = u["x_NH3_design"]["value"]       # mass fraction NH3
        self.eta_internal   = u["eta_internal"]["value"]       # fraction of Carnot
        self.k_x            = u["k_x"]["value"]                # 1/unit NH3 fraction
        self.k_T            = u["k_T"]["value"]                # 1/K condenser sensitivity
        self.PLR_min        = u["PLR_min"]["value"]
        self.plr_a          = u["plr_a"]["value"]
        self.plr_b          = u["plr_b"]["value"]
        self.plr_c          = u["plr_c"]["value"]
        self.T_approach     = u["T_approach_c"]["value"]       # degC — condenser approach

    # ------------------------------------------------------------------
    # Carnot efficiency
    # ------------------------------------------------------------------

    def eta_carnot(self, T_hot_c, T_cond_c):
        """Carnot efficiency based on hot source and condenser temperatures."""
        T_hot  = np.asarray(T_hot_c, dtype=float) + 273.15
        T_cond = np.asarray(T_cond_c, dtype=float) + 273.15
        dT = T_hot - T_cond
        return np.where(dT > 0.0, 1.0 - T_cond / T_hot, 0.0)

    # ------------------------------------------------------------------
    # Ammonia fraction correction
    # ------------------------------------------------------------------

    def f_composition(self, x_NH3):
        """
        Efficiency correction for ammonia mass fraction deviation.
        f_x = 1 + k_x * (x - x_design)
        Positive k_x: higher NH3 improves efficiency up to design point.
        Valid ±0.1 around design.
        """
        x = np.asarray(x_NH3, dtype=float)
        f = 1.0 + self.k_x * (x - self.x_NH3_design)
        return np.clip(f, 0.6, 1.3)

    # ------------------------------------------------------------------
    # Condenser temperature correction
    # ------------------------------------------------------------------

    def f_condenser(self, T_cond_c):
        """
        Efficiency correction for condenser temperature.
        Ammonia has very high condensing pressure sensitivity (dp/dT large).
        f_T = 1 - k_T * (T_cond - T_cond_design)
        """
        T = np.asarray(T_cond_c, dtype=float)
        f = 1.0 - self.k_T * (T - self.T_cond_design)
        return np.clip(f, 0.3, 1.4)

    # ------------------------------------------------------------------
    # Part-load correction
    # ------------------------------------------------------------------

    def f_plr(self, PLR):
        """Quadratic part-load efficiency correction."""
        PLR = np.asarray(PLR, dtype=float)
        PLR_eff = np.maximum(PLR, self.PLR_min)
        return self.plr_a + self.plr_b * PLR_eff + self.plr_c * PLR_eff**2

    # ------------------------------------------------------------------
    # Net efficiency
    # ------------------------------------------------------------------

    def efficiency(self, T_hot_c, T_cond_c, PLR=1.0, x_NH3=None):
        """
        Net Kalina cycle thermal efficiency.
        eta = eta_Carnot * eta_internal * f_x * f_T * f_PLR
        """
        if x_NH3 is None:
            x_NH3 = self.x_NH3_design
        eta_c  = self.eta_carnot(T_hot_c, T_cond_c)
        eta_d  = eta_c * self.eta_internal
        f_x    = self.f_composition(x_NH3)
        f_t    = self.f_condenser(T_cond_c)
        f_p    = self.f_plr(PLR)
        eta    = eta_d * f_x * f_t * f_p
        # Must not exceed Carnot
        return np.clip(eta, 0.0, eta_c)

    # ------------------------------------------------------------------
    # Power and heat flows
    # ------------------------------------------------------------------

    def _heat_input(self, T_hot_c, T_cond_c, PLR, heat_input_kw):
        """Resolve heat input from rated conditions or user-supplied."""
        PLR_arr = np.asarray(PLR, dtype=float)
        if heat_input_kw is not None:
            return np.asarray(heat_input_kw, dtype=float)
        eta_r = float(self.efficiency(self.T_hot_design, self.T_cond_design, 1.0))
        Q_hot_rated = self.P_rated / max(eta_r, 1e-6)
        return Q_hot_rated * PLR_arr

    def power_output_kw(self, T_hot_c, T_cond_c, PLR=1.0,
                         x_NH3=None, heat_input_kw=None):
        """Net electrical output [kW]."""
        eta   = self.efficiency(T_hot_c, T_cond_c, PLR, x_NH3)
        Q_hot = self._heat_input(T_hot_c, T_cond_c, PLR, heat_input_kw)
        return Q_hot * eta

    def heat_rejection_kw(self, T_hot_c, T_cond_c, PLR=1.0,
                           x_NH3=None, heat_input_kw=None):
        """Heat rejected to condenser [kW]."""
        Q_hot = self._heat_input(T_hot_c, T_cond_c, PLR, heat_input_kw)
        P_out = self.power_output_kw(T_hot_c, T_cond_c, PLR, x_NH3, heat_input_kw)
        return Q_hot - P_out

    # ------------------------------------------------------------------
    # predict_all
    # ------------------------------------------------------------------

    def predict_all(self, T_hot_c, T_cond_c, PLR=1.0,
                     x_NH3=None, heat_input_kw=None):
        """
        Return all outputs as a dict.

        Parameters
        ----------
        T_hot_c       : heat source temperature [degC]
        T_cond_c      : condenser temperature [degC]
        PLR           : part-load ratio [0.3-1.0]
        x_NH3         : ammonia mass fraction [0.7-1.0] (optional, uses design if None)
        heat_input_kw : heat input [kW] (optional; back-calculated from rated if None)
        """
        if x_NH3 is None:
            x_NH3 = self.x_NH3_design
        eta   = self.efficiency(T_hot_c, T_cond_c, PLR, x_NH3)
        eta_c = self.eta_carnot(T_hot_c, T_cond_c)
        P_out = self.power_output_kw(T_hot_c, T_cond_c, PLR, x_NH3, heat_input_kw)
        Q_rej = self.heat_rejection_kw(T_hot_c, T_cond_c, PLR, x_NH3, heat_input_kw)
        f_x   = self.f_composition(x_NH3)
        f_t   = self.f_condenser(T_cond_c)

        return {
            "efficiency":          eta,
            "eta_carnot":          eta_c,
            "power_output_kw":     P_out,
            "heat_rejection_kw":   Q_rej,
            "f_composition":       np.asarray(f_x, dtype=float),
            "f_condenser":         np.asarray(f_t, dtype=float),
        }
