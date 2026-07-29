"""
EC097 — Rankine Cycle (Steam Turbine) — F1a Efficiency Curve

Part-load efficiency model:
    eta = eta_rated * (1 - a*(1-PLR)^2)
    PLR = P / P_rated

with Carnot limit check based on steam/condenser temperatures.

Reference:
    Cotton (1998). Evaluating and Improving Steam Turbine Performance.
    Lior (2002). "Power from steam." Energy Conversion and Management.
"""

import numpy as np


class RankineCycleF1a:
    """Rankine cycle steam turbine — part-load efficiency curve model."""

    def __init__(self, params: dict):
        t = params["turbine"]
        self.P_rated = t["P_rated"]["value"]           # W
        self.eta_rated = t["eta_rated"]["value"]        # -
        self.a_partload = t["a_partload"]["value"]      # -
        self.T_steam = t["T_steam"]["value"]            # degC
        self.P_steam = t["P_steam"]["value"]            # bar
        self.T_condenser = t["T_condenser"]["value"]    # degC
        self.PLR_min = t["PLR_min"]["value"]            # -

    def carnot_efficiency(self, T_hot_c, T_cold_c):
        """Carnot efficiency as upper bound."""
        T_hot = np.asarray(T_hot_c, dtype=float) + 273.15
        T_cold = np.asarray(T_cold_c, dtype=float) + 273.15
        return 1.0 - T_cold / T_hot

    def cycle_efficiency(self, PLR, T_steam_c=None, T_cond_c=None):
        """
        Cycle efficiency at given part-load ratio.

        Args:
            PLR: Part-load ratio P/P_rated (0-1)
            T_steam_c: Steam temperature in degC (optional, default from params)
            T_cond_c: Condenser temperature in degC (optional, default from params)

        Returns:
            Cycle efficiency (dimensionless)
        """
        PLR = np.asarray(PLR, dtype=float)
        T_s = self.T_steam if T_steam_c is None else np.asarray(T_steam_c, dtype=float)
        T_c = self.T_condenser if T_cond_c is None else np.asarray(T_cond_c, dtype=float)

        # Part-load efficiency
        eta = self.eta_rated * (1.0 - self.a_partload * (1.0 - PLR) ** 2)

        # Carnot limit
        eta_carnot = self.carnot_efficiency(T_s, T_c)
        eta = np.minimum(eta, eta_carnot)

        # Below minimum load, efficiency drops to zero
        eta = np.where(PLR < self.PLR_min, 0.0, eta)
        eta = np.where(PLR <= 0.0, 0.0, eta)

        return np.clip(eta, 0.0, 1.0)

    def power_output(self, PLR):
        """Electrical power output in W."""
        PLR = np.asarray(PLR, dtype=float)
        return np.clip(PLR, 0.0, 1.0) * self.P_rated

    def heat_input(self, PLR, T_steam_c=None, T_cond_c=None):
        """Thermal heat input in W."""
        PLR = np.asarray(PLR, dtype=float)
        P_out = self.power_output(PLR)
        eta = self.cycle_efficiency(PLR, T_steam_c, T_cond_c)
        return np.where(eta > 0, P_out / eta, 0.0)

    def steam_mass_flow(self, PLR, T_steam_c=None, T_cond_c=None):
        """
        Approximate steam mass flow rate in kg/s.
        Uses enthalpy drop estimate: ~1000 kJ/kg for subcritical steam.
        """
        Q_in = self.heat_input(PLR, T_steam_c, T_cond_c)
        h_drop = 1000e3  # J/kg approximate enthalpy drop
        return Q_in / h_drop
