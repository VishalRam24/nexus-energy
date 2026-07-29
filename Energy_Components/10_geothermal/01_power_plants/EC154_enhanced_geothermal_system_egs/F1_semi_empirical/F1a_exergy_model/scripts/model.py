"""
EC154 — Enhanced Geothermal System (EGS) — F1a Exergy Efficiency Model

EGS extracts heat from hot dry rock by:
  1. Hydraulic fracturing to create a subsurface heat exchanger
  2. Circulating water through the fracture network
  3. Converting heat via binary/ORC cycle at surface

Key difference from natural geothermal:
  - Requires significant circulation pump work (parasitic load)
  - Lower eta_utilization due to pump parasitics and fracture heterogeneity
  - Applicable T range: 150–350°C (depth ~3–10 km)

Model:
    eta_Carnot   = 1 - T_reinject / T_geo  (K)
    eta_gross    = eta_util * eta_Carnot
    P_gross      = m_dot * cp * (T_geo - T_reinject) * eta_gross
    P_parasitic  = pump_parasitic_fraction * P_gross
    P_net        = P_gross - P_parasitic = P_gross * (1 - pump_parasitic_fraction)

Reference:
    Tester, J. et al. (2006). The Future of Geothermal Energy. MIT/DOE.
    DiPippo, R. (2015). Geothermal Power Plants, 4th ed., Chapter 16 (EGS).
"""

import numpy as np


class EGSF1a:
    """
    Enhanced Geothermal System (EGS) — exergy-based efficiency model.
    Accounts for circulation pump parasitic loads.
    """

    def __init__(self, params: dict):
        u = params["unit"]
        self.cp_fluid            = u["cp_fluid"]["value"]              # J/(kg·K)
        self.eta_util            = u["eta_utilization"]["value"]       # dimensionless
        self.pump_parasitic_frac = u["pump_parasitic_fraction"]["value"]  # dimensionless
        self.T_reinject_offset   = u["T_reinject_offset"]["value"]    # degC

    def carnot_efficiency(self, T_geo_c, T_reinject_c):
        """
        Carnot efficiency between geothermal source and reinjection sink.

        Parameters
        ----------
        T_geo_c      : float or array — rock/fluid temperature at depth (degC)
        T_reinject_c : float or array — reinjection temperature (degC)

        Returns
        -------
        eta_Carnot : ideal thermodynamic efficiency
        """
        T_geo    = np.asarray(T_geo_c, dtype=float) + 273.15
        T_reinj  = np.asarray(T_reinject_c, dtype=float) + 273.15
        eta = 1.0 - T_reinj / T_geo
        return np.clip(eta, 0.0, 1.0)

    def reinjection_temperature(self, T_reject_c):
        """Reinjection temperature (degC) = T_rejection + offset."""
        return np.asarray(T_reject_c, dtype=float) + self.T_reinject_offset

    def gross_efficiency(self, T_geo_c, T_reject_c):
        """Gross cycle efficiency before parasitic deduction."""
        T_reinj = self.reinjection_temperature(T_reject_c)
        return self.eta_util * self.carnot_efficiency(T_geo_c, T_reinj)

    def net_efficiency(self, T_geo_c, T_reject_c):
        """Net plant efficiency including pump parasitic losses."""
        eta_gross = self.gross_efficiency(T_geo_c, T_reject_c)
        return eta_gross * (1.0 - self.pump_parasitic_frac)

    def heat_input(self, T_geo_c, T_reject_c, m_dot_kgs):
        """
        Thermal energy extracted from rock (kW).
        Q = m_dot * cp * (T_geo - T_reinject)
        """
        T_geo   = np.asarray(T_geo_c, dtype=float)
        T_reinj = self.reinjection_temperature(T_reject_c)
        m_dot   = np.asarray(m_dot_kgs, dtype=float)
        dT = np.clip(T_geo - T_reinj, 0.0, None)
        return m_dot * self.cp_fluid * dT / 1000.0  # W → kW

    def gross_power(self, T_geo_c, T_reject_c, m_dot_kgs):
        """Gross electrical output before parasitic deduction (kW)."""
        Q = self.heat_input(T_geo_c, T_reject_c, m_dot_kgs)
        eta = self.gross_efficiency(T_geo_c, T_reject_c)
        return Q * eta

    def net_power(self, T_geo_c, T_reject_c, m_dot_kgs):
        """
        Net electrical power output after pump parasitic deduction (kW).
        P_net = P_gross * (1 - pump_parasitic_fraction)
        """
        P_gross = self.gross_power(T_geo_c, T_reject_c, m_dot_kgs)
        return P_gross * (1.0 - self.pump_parasitic_frac)

    def parasitic_power(self, T_geo_c, T_reject_c, m_dot_kgs):
        """Circulation pump electrical consumption (kW)."""
        P_gross = self.gross_power(T_geo_c, T_reject_c, m_dot_kgs)
        return P_gross * self.pump_parasitic_frac
