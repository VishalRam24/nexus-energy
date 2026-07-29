"""
EC151 — Dry Steam Geothermal Plant — F1a Exergy Efficiency Model

Dry steam plants pipe steam directly from the reservoir to the turbine —
the simplest and oldest geothermal power plant type (e.g. The Geysers, CA).

eta_Carnot = 1 - T_condenser / T_geo  (temperatures in Kelvin)
eta_plant  = eta_utilization * eta_Carnot
P_net      = m_dot * cp_steam * (T_geo - T_condenser) * eta_plant

T_geo range: 180–280 °C (must deliver dry steam at wellhead)
eta_thermal: typically 15–21% (higher than binary because direct cycle)
eta_exergy:  typically 50–65% of Carnot

Reference:
    DiPippo, R. (2015). Geothermal Power Plants: Principles, Applications,
    Case Studies and Environmental Impact, 4th ed. Butterworth-Heinemann.
    Chapter 7 — Dry Steam Power Plants.
"""

import numpy as np


class DrySteamGeothermalF1a:
    """
    Dry steam geothermal power plant — exergy-based efficiency model.
    Steam is delivered directly from the reservoir to the turbine.
    """

    def __init__(self, params: dict):
        u = params["unit"]
        self.cp_steam = u["cp_steam"]["value"]                      # J/(kg·K)
        self.eta_util = u["eta_utilization"]["value"]               # dimensionless
        self.T_condenser_offset = u["T_condenser_offset"]["value"]  # degC

    def carnot_efficiency(self, T_geo_c, T_reject_c):
        """
        Carnot efficiency between geothermal steam source and condenser sink.

        Parameters
        ----------
        T_geo_c    : float or array  — steam wellhead temperature (degC)
        T_reject_c : float or array  — condenser cooling rejection temperature (degC)

        Returns
        -------
        eta_Carnot : float or array  — ideal thermodynamic efficiency
        """
        T_geo = np.asarray(T_geo_c, dtype=float) + 273.15
        T_cond = np.asarray(T_reject_c, dtype=float) + self.T_condenser_offset + 273.15
        eta = 1.0 - T_cond / T_geo
        return np.clip(eta, 0.0, 1.0)

    def plant_efficiency(self, T_geo_c, T_reject_c):
        """Overall plant efficiency = eta_util * eta_Carnot."""
        return self.eta_util * self.carnot_efficiency(T_geo_c, T_reject_c)

    def condenser_temperature(self, T_reject_c):
        """Condenser saturation temperature (degC)."""
        return np.asarray(T_reject_c, dtype=float) + self.T_condenser_offset

    def heat_input(self, T_geo_c, T_reject_c, m_dot_kgs):
        """
        Thermal energy extracted from steam (kW).
        Q = m_dot * cp_steam * (T_geo - T_condenser)
        """
        T_geo  = np.asarray(T_geo_c, dtype=float)
        T_cond = self.condenser_temperature(T_reject_c)
        m_dot  = np.asarray(m_dot_kgs, dtype=float)
        dT = np.clip(T_geo - T_cond, 0.0, None)
        return m_dot * self.cp_steam * dT / 1000.0  # W → kW

    def power_output(self, T_geo_c, T_reject_c, m_dot_kgs):
        """
        Net electrical power output (kW).
        P = Q_heat * eta_plant
        """
        Q = self.heat_input(T_geo_c, T_reject_c, m_dot_kgs)
        eta = self.plant_efficiency(T_geo_c, T_reject_c)
        return Q * eta
