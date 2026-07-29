"""
EC058 — Flat Plate Solar Collector — F1b IAM Model

Hottel-Whillier-Bliss equation with Incidence Angle Modifier:
    IAM(theta) = 1 - b0 * (1/cos(theta) - 1)
    Q_u = A * F_R * [IAM * tau_alpha * G - U_L * (T_in - T_amb)]
    eta = Q_u / (A * G)
    T_out = T_in + Q_u / (m_dot * cp)

Improvement over F1a:
    - Adds angular dependence of optical efficiency via IAM
    - Separates F_R and tau_alpha (F1a lumped them as F_R*tau_alpha)
    - More accurate for off-normal incidence common in real installations

References:
    Duffie & Beckman (2013), 'Solar Engineering of Thermal Processes',
    4th ed., Wiley, Ch. 6.
    ASHRAE Standard 93 (2010) — Methods of Testing to Determine the
    Thermal Performance of Solar Collectors.
"""

import numpy as np


class FlatPlateCollectorF1b:
    """Flat-plate collector with incidence angle modifier (IAM)."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.area = u["area"]["value"]           # m2
        self.F_R = u["F_R"]["value"]             # dimensionless
        self.tau_alpha = u["tau_alpha"]["value"]  # at normal incidence
        self.U_L = u["U_L"]["value"]             # W/m2K
        self.b0 = u["b0"]["value"]               # IAM coefficient
        self.m_dot = u["flow_rate"]["value"]     # kg/s
        self.cp = u["cp"]["value"]               # J/kgK

    def iam(self, theta_deg):
        """
        Incidence Angle Modifier.
        IAM(theta) = 1 - b0 * (1/cos(theta) - 1)
        Valid for theta < 80 deg; returns 0 beyond.
        """
        theta = np.asarray(theta_deg, dtype=float)
        theta_rad = np.radians(theta)
        cos_theta = np.cos(theta_rad)
        # Avoid division by zero and extreme values
        safe_cos = np.where(theta < 80.0, np.maximum(cos_theta, 0.01), 0.01)
        iam_val = 1.0 - self.b0 * (1.0 / safe_cos - 1.0)
        iam_val = np.where(theta < 80.0, iam_val, 0.0)
        return np.clip(iam_val, 0.0, 1.0)

    def useful_heat_w(self, irradiance, incidence_angle_deg, T_inlet_c, T_ambient_c):
        """
        Useful heat gain in Watts.

        Parameters
        ----------
        irradiance         : solar irradiance on collector plane (W/m2)
        incidence_angle_deg: angle of incidence (degrees)
        T_inlet_c          : fluid inlet temperature (degC)
        T_ambient_c        : ambient temperature (degC)
        """
        G = np.asarray(irradiance, dtype=float)
        T_in = np.asarray(T_inlet_c, dtype=float)
        T_amb = np.asarray(T_ambient_c, dtype=float)

        iam_val = self.iam(incidence_angle_deg)

        Q_u = self.area * self.F_R * (iam_val * self.tau_alpha * G
                                       - self.U_L * (T_in - T_amb))
        return np.maximum(0.0, Q_u)

    def efficiency(self, irradiance, incidence_angle_deg, T_inlet_c, T_ambient_c):
        """Instantaneous collector efficiency eta = Q_u / (A * G)."""
        G = np.asarray(irradiance, dtype=float)
        Q_u = self.useful_heat_w(irradiance, incidence_angle_deg, T_inlet_c, T_ambient_c)
        denom = self.area * np.where(G > 1.0, G, 1.0)
        eta = np.where(G > 1.0, Q_u / denom, 0.0)
        return np.clip(eta, 0.0, 1.0)

    def T_outlet(self, irradiance, incidence_angle_deg, T_inlet_c, T_ambient_c):
        """Outlet temperature (steady-state)."""
        Q_u = self.useful_heat_w(irradiance, incidence_angle_deg, T_inlet_c, T_ambient_c)
        T_in = np.asarray(T_inlet_c, dtype=float)
        return T_in + Q_u / (self.m_dot * self.cp)

    def predict_all(self, irradiance, incidence_angle_deg, T_inlet_c, T_ambient_c):
        """Return all outputs."""
        Q_u = self.useful_heat_w(irradiance, incidence_angle_deg, T_inlet_c, T_ambient_c)
        eta = self.efficiency(irradiance, incidence_angle_deg, T_inlet_c, T_ambient_c)
        iam_val = self.iam(incidence_angle_deg)
        T_out = self.T_outlet(irradiance, incidence_angle_deg, T_inlet_c, T_ambient_c)
        return {
            "thermal_output_w": Q_u,
            "efficiency": eta,
            "iam_factor": iam_val,
            "T_outlet_degC": T_out,
        }
