"""
EC055 — Solar Tower Central Receiver — F1a Optical + Thermal Efficiency Model

Heliostat-field optical efficiency:
    eta_field(theta_z) = eta_field_peak * f_cos(theta_z) * f_atm(theta_z)
    f_cos(theta_z)     = 1 - cosine_loss_coeff * theta_z^2     (cosine + spillage proxy)
    f_atm(theta_z)     = 1 - atm_atten_coeff * theta_z         (atmospheric attenuation)

Power onto receiver aperture:
    Q_field = DNI * A_field * eta_field(theta_z)

Receiver thermal losses:
    Q_rad  = epsilon * sigma * A_recv * (T_recv^4 - T_amb^4)
    Q_conv = h_conv  * A_recv * (T_recv  - T_amb)
    Q_loss = Q_rad + Q_conv

Receiver absorbed (after surface absorptivity) and useful heat:
    Q_abs    = alpha * Q_field
    Q_useful = max(0, Q_abs - Q_loss)

Efficiencies:
    eta_optical  = Q_field / (DNI * A_field)
    eta_receiver = Q_useful / Q_field
    eta_overall  = Q_useful / (DNI * A_field)

Reference:
    Wagner & Wendelin (2018), 'SolarPILOT: A power tower solar field layout
    and characterization tool', Solar Energy, 171, 185-196.
    Falcone (1986), 'A handbook for solar central receiver design',
    SAND86-8009, Sandia National Laboratories.
"""

import numpy as np


class SolarTowerF1a:
    """Solar tower CSP — heliostat field optical + receiver thermal model."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.A_field = u["A_field"]["value"]
        self.eta_field_peak = u["eta_field_peak"]["value"]
        self.atm_atten = u["atm_atten_coeff"]["value"]
        self.cos_coeff = u["cosine_loss_coeff"]["value"]

        self.A_recv = u["A_receiver"]["value"]
        self.alpha = u["absorptivity"]["value"]
        self.eps = u["emissivity"]["value"]
        self.h_conv = u["h_conv"]["value"]
        self.sigma = u["stefan_boltzmann"]["value"]

    # ------------------------------------------------------------------
    # Optical
    # ------------------------------------------------------------------
    def field_efficiency(self, solar_zenith_deg):
        z = np.asarray(solar_zenith_deg, dtype=float)
        f_cos = np.clip(1.0 - self.cos_coeff * z * z, 0.0, 1.0)
        f_atm = np.clip(1.0 - self.atm_atten * z, 0.0, 1.0)
        return np.clip(self.eta_field_peak * f_cos * f_atm, 0.0, 1.0)

    def Q_field_kw(self, dni, solar_zenith_deg):
        G = np.asarray(dni, dtype=float)
        eta = self.field_efficiency(solar_zenith_deg)
        return G * self.A_field * eta / 1000.0  # kW

    # ------------------------------------------------------------------
    # Receiver thermal losses
    # ------------------------------------------------------------------
    def thermal_loss_kw(self, T_recv_c, T_amb_c):
        T_r = np.asarray(T_recv_c, dtype=float) + 273.15
        T_a = np.asarray(T_amb_c, dtype=float) + 273.15
        Q_rad = self.eps * self.sigma * self.A_recv * (T_r ** 4 - T_a ** 4)
        Q_conv = self.h_conv * self.A_recv * (T_r - T_a)
        return (Q_rad + Q_conv) / 1000.0  # kW

    def predict_all(self, dni, solar_zenith_deg, T_recv_c, T_amb_c):
        Q_field = self.Q_field_kw(dni, solar_zenith_deg)
        Q_abs = self.alpha * Q_field
        Q_loss = self.thermal_loss_kw(T_recv_c, T_amb_c)
        Q_useful = np.maximum(0.0, Q_abs - Q_loss)

        G = np.asarray(dni, dtype=float)
        P_inc_safe = np.where(G > 0.01, G * self.A_field / 1000.0, 1.0)  # kW
        eta_opt = np.where(G > 0.01, Q_field / P_inc_safe, 0.0)
        eta_recv = np.where(Q_field > 1e-6, Q_useful / np.maximum(Q_field, 1e-6), 0.0)
        eta_overall = np.where(G > 0.01, Q_useful / P_inc_safe, 0.0)

        return {
            "Q_field_kw": Q_field,
            "useful_heat_kw": Q_useful,
            "thermal_loss_kw": Q_loss,
            "optical_efficiency": np.clip(eta_opt, 0.0, 1.0),
            "receiver_efficiency": np.clip(eta_recv, 0.0, 1.0),
            "overall_efficiency": np.clip(eta_overall, 0.0, 1.0),
        }
