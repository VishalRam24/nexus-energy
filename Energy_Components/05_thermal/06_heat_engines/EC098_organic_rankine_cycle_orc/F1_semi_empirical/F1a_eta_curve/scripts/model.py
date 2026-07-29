"""
EC098 — Organic Rankine Cycle (ORC) — F1a Efficiency Curve Model

eta_thermal = eta_Carnot * eta_internal * f_PLR

where:
  eta_Carnot   = 1 - T_cold_K / T_hot_K          (ideal upper bound)
  eta_internal = eta_expander * eta_pump           (irreversibilities in cycle)
  f_PLR        = c0 + c1 * PLR                    (linear part-load correction)
  P_out        = Q_hot * eta_thermal
  Q_reject     = Q_hot - P_out

References:
    Quoilin, Van Den Broek, Declaye, Dewallef & Lemort (2013),
    'Techno-economic survey of Organic Rankine Cycle (ORC) systems',
    Ren. Sustain. Energy Rev., 22, 168-186.
"""

import numpy as np


class ORCF1a:
    """Organic Rankine Cycle — efficiency as f(T_hot, T_cold, PLR)."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.P_rated = u["P_rated"]["value"]          # kW_e
        self.eta_internal = u["eta_internal"]["value"] # dimensionless
        self.f_PLR_c0 = u["f_PLR_c0"]["value"]        # part-load intercept
        self.f_PLR_c1 = u["f_PLR_c1"]["value"]        # part-load slope

    def eta_carnot(self, T_hot_c, T_cold_c):
        """Carnot efficiency (upper bound)."""
        T_hot = np.asarray(T_hot_c, dtype=float) + 273.15
        T_cold = np.asarray(T_cold_c, dtype=float) + 273.15
        # Guard: T_hot must exceed T_cold
        dT = T_hot - T_cold
        return np.where(dT > 0.0, 1.0 - T_cold / T_hot, 0.0)

    def part_load_factor(self, plr):
        """Linear part-load correction factor."""
        plr = np.asarray(plr, dtype=float)
        return self.f_PLR_c0 + self.f_PLR_c1 * plr

    def efficiency(self, T_hot_c, T_cold_c, plr=1.0):
        """Net thermal efficiency of the ORC cycle."""
        eta_c = self.eta_carnot(T_hot_c, T_cold_c)
        f = self.part_load_factor(plr)
        eta = eta_c * self.eta_internal * f
        # Physical cap: must be less than Carnot, positive
        return np.clip(eta, 0.0, eta_c)

    def power_flows(self, T_hot_c, T_cold_c, plr=1.0, Q_hot_kw=None):
        """
        Compute power output and heat flows.

        Parameters
        ----------
        T_hot_c  : heat source temperature (degC)
        T_cold_c : heat sink temperature (degC)
        plr      : part-load ratio (0.3 – 1.0)
        Q_hot_kw : heat input (kW); if None, inferred from rated capacity at PLR=1

        Returns
        -------
        dict: efficiency, power_kw, heat_input_kw, heat_rejection_kw
        """
        eta = self.efficiency(T_hot_c, T_cold_c, plr)
        plr_arr = np.asarray(plr, dtype=float)
        if Q_hot_kw is None:
            # At rated conditions eta_rated ~ eta_Carnot * eta_internal * f_PLR(1)
            # Use rated power to back-calculate rated Q_hot
            eta_rated = self.efficiency(150.0, 30.0, 1.0)
            Q_hot_rated = self.P_rated / max(float(eta_rated), 1e-6)
            Q_hot = np.broadcast_to(Q_hot_rated * plr_arr, np.broadcast_shapes(np.shape(eta), np.shape(plr_arr)))
        else:
            Q_hot = np.asarray(Q_hot_kw, dtype=float)

        P_out = Q_hot * eta
        Q_reject = Q_hot - P_out

        return {
            "efficiency": eta,
            "eta_carnot": self.eta_carnot(T_hot_c, T_cold_c),
            "power_kw": P_out,
            "heat_input_kw": Q_hot,
            "heat_rejection_kw": Q_reject,
        }
