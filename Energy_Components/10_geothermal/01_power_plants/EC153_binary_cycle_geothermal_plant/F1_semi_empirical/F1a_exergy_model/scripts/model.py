"""
EC153 — Binary Cycle Geothermal Plant — F1a Exergy Efficiency Model

eta_plant = eta_utilization * eta_Carnot
eta_Carnot = 1 - T_reject / T_geo  (temperatures in Kelvin)
P = m_dot * cp * (T_geo - T_reinject) * eta_utilization * eta_Carnot

Reference:
    DiPippo, R. (2015). Geothermal Power Plants: Principles, Applications,
    Case Studies and Environmental Impact, 4th ed. Butterworth-Heinemann.
"""

import numpy as np


class BinaryGeothermalF1a:
    """
    Binary cycle geothermal power plant — exergy-based efficiency model.
    Predicts electrical power output and efficiency from resource conditions.
    """

    def __init__(self, params: dict):
        u = params["unit"]
        self.cp_geo = u["cp_geo"]["value"]                    # J/(kg·K)
        self.eta_util = u["eta_utilization"]["value"]         # dimensionless
        self.T_reinject_offset = u["T_reinject_offset"]["value"]  # degC

    def carnot_efficiency(self, T_geo_c, T_reject_c):
        """
        Carnot efficiency between geothermal source and rejection sink.

        Parameters
        ----------
        T_geo_c    : float or array  — geothermal temperature (degC)
        T_reject_c : float or array  — condenser / air-cooler rejection temperature (degC)

        Returns
        -------
        eta_Carnot : float or array  — ideal thermodynamic efficiency
        """
        T_geo = np.asarray(T_geo_c, dtype=float) + 273.15
        T_rej = np.asarray(T_reject_c, dtype=float) + 273.15
        eta = 1.0 - T_rej / T_geo
        return np.clip(eta, 0.0, 1.0)

    def plant_efficiency(self, T_geo_c, T_reject_c):
        """Overall plant efficiency = eta_util * eta_Carnot."""
        return self.eta_util * self.carnot_efficiency(T_geo_c, T_reject_c)

    def reinjection_temperature(self, T_reject_c):
        """Minimum reinjection temperature = T_reject + offset (degC)."""
        return np.asarray(T_reject_c, dtype=float) + self.T_reinject_offset

    def heat_input(self, T_geo_c, T_reject_c, m_dot_kgs):
        """
        Thermal energy extracted from brine (kW).
        Q = m_dot * cp * (T_geo - T_reinject)
        """
        T_geo    = np.asarray(T_geo_c, dtype=float)
        T_reinj  = self.reinjection_temperature(T_reject_c)
        m_dot    = np.asarray(m_dot_kgs, dtype=float)
        dT       = T_geo - T_reinj
        dT       = np.clip(dT, 0.0, None)
        return m_dot * self.cp_geo * dT / 1000.0  # W → kW

    def power_output(self, T_geo_c, T_reject_c, m_dot_kgs):
        """
        Net electrical power output (kW).
        P = Q_heat * eta_plant
        """
        Q = self.heat_input(T_geo_c, T_reject_c, m_dot_kgs)
        eta = self.plant_efficiency(T_geo_c, T_reject_c)
        return Q * eta
