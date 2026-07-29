"""
EC055 — Solar Tower Central Receiver — F1b With Thermal Losses

Extends F1a (optical efficiency + fixed receiver losses) with:
  1. DNI variability: optical field output scales with actual DNI
  2. T_amb variability: radiative and convective losses depend on actual T_amb
  3. Wind-dependent convective loss: h_conv = h_base + h_wind_coeff * sqrt(v_wind)
  4. Receiver efficiency as function of T_recv and T_amb

Receiver thermal loss model:
    Q_rad  = eps * sigma * A_recv * (T_recv^4 - T_amb^4)    [radiative]
    Q_conv = h_conv(v_wind) * A_recv * (T_recv - T_amb)      [convective]
    Q_loss = Q_rad + Q_conv

    Q_field  = DNI * A_field * eta_field(zenith)
    Q_abs    = absorptivity * Q_field
    Q_useful = max(0, Q_abs - Q_loss)

    eta_receiver = Q_useful / Q_field     (0 when Q_field = 0)
    eta_overall  = Q_useful / (DNI * A_field)

Improvement over F1a:
    - F1a used a fixed ambient temperature; F1b accepts T_amb as input
    - F1b adds wind-dependent convective term
    - F1b enables sensitivity to T_amb and wind for dispatch optimization

References:
    Wagner & Wendelin (2018). Solar Energy 171, 185-196.
    Falcone (1986). SAND86-8009, Sandia National Labs.
    Kolb (2011). SAND2011-2419, Sandia — Gemasolar performance analysis.
    Siebers & Kraabel (1984). SAND84-8717, convective loss correlations.
"""

import numpy as np


class SolarTowerF1b:
    """Solar tower — heliostat field + explicit thermal loss model (T_amb, wind)."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.A_field = u["A_field"]["value"]
        self.eta_field_peak = u["eta_field_peak"]["value"]
        self.atm_atten = u["atm_atten_coeff"]["value"]
        self.cos_coeff = u["cosine_loss_coeff"]["value"]

        self.A_recv = u["A_receiver"]["value"]
        self.alpha = u["absorptivity"]["value"]
        self.eps = u["emissivity"]["value"]
        self.h_base = u["h_conv_base"]["value"]
        self.h_wind_coeff = u["h_conv_wind_coeff"]["value"]
        self.sigma = u["stefan_boltzmann"]["value"]

    # ------------------------------------------------------------------
    # Optical
    # ------------------------------------------------------------------
    def field_efficiency(self, solar_zenith_deg):
        z = np.asarray(solar_zenith_deg, dtype=float)
        f_cos = np.clip(1.0 - self.cos_coeff * z * z, 0.0, 1.0)
        f_atm = np.clip(1.0 - self.atm_atten * z, 0.0, 1.0)
        return np.clip(self.eta_field_peak * f_cos * f_atm, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Convective heat transfer coefficient
    # ------------------------------------------------------------------
    def h_conv(self, wind_speed_m_s=0.0):
        """
        Wind-dependent convective HT coefficient.
        h = h_base + h_wind_coeff * sqrt(v_wind)
        Based on Siebers & Kraabel (1984) correlation for cylindrical receivers.
        """
        v = np.asarray(wind_speed_m_s, dtype=float)
        return self.h_base + self.h_wind_coeff * np.sqrt(np.maximum(v, 0.0))

    # ------------------------------------------------------------------
    # Thermal losses
    # ------------------------------------------------------------------
    def thermal_loss_kw(self, T_recv_c, T_amb_c, wind_speed_m_s=0.0):
        """
        Total receiver thermal loss (radiative + convective).
        Q_rad  = eps * sigma * A * (T_r^4 - T_a^4)
        Q_conv = h(v_wind) * A * (T_r - T_a)
        """
        T_r = np.asarray(T_recv_c, dtype=float) + 273.15
        T_a = np.asarray(T_amb_c, dtype=float) + 273.15
        h = self.h_conv(wind_speed_m_s)

        Q_rad = self.eps * self.sigma * self.A_recv * (T_r ** 4 - T_a ** 4)
        Q_conv = h * self.A_recv * (T_r - T_a)
        return (Q_rad + Q_conv) / 1000.0  # kW

    # ------------------------------------------------------------------
    # Predict all
    # ------------------------------------------------------------------
    def predict_all(self, dni, solar_zenith_deg, T_recv_c, T_amb_c, wind_speed_m_s=0.0):
        """
        Parameters
        ----------
        dni             : W/m2, direct normal irradiance
        solar_zenith_deg: degrees, solar zenith angle
        T_recv_c        : degC, receiver surface temperature
        T_amb_c         : degC, ambient temperature
        wind_speed_m_s  : m/s, wind speed at receiver height (default 0)
        """
        G = np.asarray(dni, dtype=float)
        eta_field = self.field_efficiency(solar_zenith_deg)
        Q_field = G * self.A_field * eta_field / 1000.0  # kW

        Q_abs = self.alpha * Q_field
        Q_loss = self.thermal_loss_kw(T_recv_c, T_amb_c, wind_speed_m_s)
        Q_useful = np.maximum(0.0, Q_abs - Q_loss)

        P_inc = np.where(G > 0.01, G * self.A_field / 1000.0, 1.0)  # kW
        eta_opt = np.where(G > 0.01, Q_field / P_inc, 0.0)
        eta_recv = np.where(Q_field > 1e-3, Q_useful / np.maximum(Q_field, 1e-3), 0.0)
        eta_overall = np.where(G > 0.01, Q_useful / P_inc, 0.0)

        return {
            "Q_field_kw": Q_field,
            "useful_heat_kw": Q_useful,
            "thermal_loss_kw": Q_loss,
            "Q_radiative_kw": self.eps * self.sigma * self.A_recv
                              * ((np.asarray(T_recv_c) + 273.15) ** 4
                                 - (np.asarray(T_amb_c) + 273.15) ** 4) / 1000.0,
            "Q_convective_kw": self.h_conv(wind_speed_m_s) * self.A_recv
                               * (np.asarray(T_recv_c) - np.asarray(T_amb_c)) / 1000.0,
            "optical_efficiency": np.clip(eta_opt, 0.0, 1.0),
            "receiver_efficiency": np.clip(eta_recv, 0.0, 1.0),
            "overall_efficiency": np.clip(eta_overall, 0.0, 1.0),
            "h_conv_w_m2k": np.asarray(self.h_conv(wind_speed_m_s) * np.ones_like(G)),
        }
