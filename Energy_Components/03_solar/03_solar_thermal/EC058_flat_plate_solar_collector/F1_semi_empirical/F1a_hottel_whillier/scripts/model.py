"""
EC058 — Flat Plate Solar Collector — F1a Hottel-Whillier-Bliss Model

Hottel-Whillier-Bliss (HWB) equation:
    Q_u = A * [F_R*(tau*alpha) * G - F_R*U_L * (T_in - T_amb)]
    eta  = Q_u / (A * G)
         = F_R*(tau*alpha) - F_R*U_L * (T_in - T_amb) / G

Collector cannot extract negative heat:
    Q_u = max(0, Q_u)

Outlet temperature approximation (assuming steady state, no heat capacity):
    T_out = T_in + Q_u / (m_dot * cp)

References:
    Duffie & Beckman (2013), 'Solar Engineering of Thermal Processes',
    4th ed., John Wiley & Sons, Ch. 6.
"""

import numpy as np


class FlatPlateCollectorF1a:
    """Flat-plate solar collector — Hottel-Whillier-Bliss steady-state model."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.area = u["area"]["value"]               # m2
        self.FR_tau_alpha = u["F_R_tau_alpha"]["value"]  # dimensionless
        self.FR_U_L = u["F_R_U_L"]["value"]          # W/m2K
        self.m_dot = u["m_dot"]["value"]             # kg/s
        self.cp = u["cp_fluid"]["value"]             # J/kgK

    def useful_heat_w(self, irradiance, T_inlet_c, T_ambient_c):
        """
        Useful heat gain in Watts.

        Parameters
        ----------
        irradiance  : solar irradiance on collector plane (W/m2)
        T_inlet_c   : fluid inlet temperature (degC)
        T_ambient_c : ambient temperature (degC)
        """
        G = np.asarray(irradiance, dtype=float)
        T_in = np.asarray(T_inlet_c, dtype=float)
        T_amb = np.asarray(T_ambient_c, dtype=float)

        Q_u = self.area * (self.FR_tau_alpha * G - self.FR_U_L * (T_in - T_amb))
        return np.maximum(0.0, Q_u)   # collector cannot cool fluid below ambient driving

    def efficiency(self, irradiance, T_inlet_c, T_ambient_c):
        """
        Instantaneous collector efficiency eta = Q_u / (A * G).
        Returns 0 when G == 0 to avoid divide-by-zero.
        """
        G = np.asarray(irradiance, dtype=float)
        Q_u = self.useful_heat_w(irradiance, T_inlet_c, T_ambient_c)
        denom = self.area * np.where(G > 1.0, G, 1.0)   # safe denominator
        eta = np.where(G > 1.0, Q_u / denom, 0.0)
        return np.clip(eta, 0.0, self.FR_tau_alpha)

    def T_outlet(self, irradiance, T_inlet_c, T_ambient_c):
        """Approximate outlet temperature (steady-state, constant m_dot)."""
        Q_u = self.useful_heat_w(irradiance, T_inlet_c, T_ambient_c)
        T_in = np.asarray(T_inlet_c, dtype=float)
        return T_in + Q_u / (self.m_dot * self.cp)

    def predict_all(self, irradiance, T_inlet_c, T_ambient_c):
        """Return all outputs as a dict."""
        Q_u = self.useful_heat_w(irradiance, T_inlet_c, T_ambient_c)
        eta = self.efficiency(irradiance, T_inlet_c, T_ambient_c)
        T_out = self.T_outlet(irradiance, T_inlet_c, T_ambient_c)
        return {
            "useful_heat_w": Q_u,
            "efficiency": eta,
            "T_outlet_c": T_out,
        }
