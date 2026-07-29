"""
EC059 — Evacuated Tube Solar Collector — F1a Hottel-Whillier-Bliss Model

Hottel-Whillier-Bliss (HWB) equation:
    Q_u = A * [F_R*(tau*alpha) * G - F_R*U_L * (T_in - T_amb)]
    eta = Q_u / (A * G) = F_R*(tau*alpha) - F_R*U_L * (T_in - T_amb) / G

Evacuated tube specifics:
    - F_R*U_L is very low (~1-2 W/m2K) due to vacuum insulation between
      absorber tube and outer glass tube. Flat-plate is typically ~4-6 W/m2K.
    - This makes ETCs the preferred choice when (T_in - T_amb)/G is large
      (winter, high-T process heat, low solar fraction).
    - F_R*(tau*alpha) is similar to or slightly lower than flat plate.

Outlet temperature (steady-state):
    T_out = T_in + Q_u / (m_dot * cp)

Reference:
    Duffie & Beckman (2013), 'Solar Engineering of Thermal Processes',
    4th ed., John Wiley & Sons, Ch. 6.
    SRCC OG-100 ETC certification reports.
"""

import numpy as np


class EvacuatedTubeF1a:
    """Evacuated tube solar collector — Hottel-Whillier-Bliss steady-state model."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.area = u["area"]["value"]
        self.FR_tau_alpha = u["F_R_tau_alpha"]["value"]
        self.FR_U_L = u["F_R_U_L"]["value"]
        self.m_dot = u["m_dot"]["value"]
        self.cp = u["cp_fluid"]["value"]

    def useful_heat_w(self, irradiance, T_inlet_c, T_ambient_c):
        G = np.asarray(irradiance, dtype=float)
        T_in = np.asarray(T_inlet_c, dtype=float)
        T_amb = np.asarray(T_ambient_c, dtype=float)
        Q_u = self.area * (self.FR_tau_alpha * G - self.FR_U_L * (T_in - T_amb))
        return np.maximum(0.0, Q_u)

    def efficiency(self, irradiance, T_inlet_c, T_ambient_c):
        G = np.asarray(irradiance, dtype=float)
        Q_u = self.useful_heat_w(irradiance, T_inlet_c, T_ambient_c)
        denom = self.area * np.where(G > 1.0, G, 1.0)
        eta = np.where(G > 1.0, Q_u / denom, 0.0)
        return np.clip(eta, 0.0, self.FR_tau_alpha)

    def T_outlet(self, irradiance, T_inlet_c, T_ambient_c):
        Q_u = self.useful_heat_w(irradiance, T_inlet_c, T_ambient_c)
        T_in = np.asarray(T_inlet_c, dtype=float)
        return T_in + Q_u / (self.m_dot * self.cp)

    def predict_all(self, irradiance, T_inlet_c, T_ambient_c):
        Q_u = self.useful_heat_w(irradiance, T_inlet_c, T_ambient_c)
        eta = self.efficiency(irradiance, T_inlet_c, T_ambient_c)
        T_out = self.T_outlet(irradiance, T_inlet_c, T_ambient_c)
        return {
            "useful_heat_w": Q_u,
            "efficiency": eta,
            "T_outlet_c": T_out,
        }
