"""
EC061 — Unglazed Solar Collector (Pool Heating) — F1b Incidence Angle Modifier

Unglazed collectors (EPDM/PE mats) are used primarily for pool heating at low
temperature lifts (ΔT < 15 K). No glass cover means no transmitted-reflectance IAM.

Extends F1a (basic Hottel-Whillier) by adding:
  1. ASHRAE 93-based IAM: b0 biaxial modifier for polymer mat surface
  2. Wind-speed correction on heat loss coefficient (important for unglazed)
  3. Sky radiation loss term (unglazed sees cold sky directly)

Extended HWB for unglazed with IAM and wind-corrected U_L:
    U_L(v_wind) = U_L_0 + U_wind * v_wind
    IAM(theta) = 1 - b0 * (1/cos(theta) - 1)   [ASHRAE 93 model]
    Q_sky = eps * sigma * A * (T_col^4 - T_sky^4)   [net sky radiation loss]
    Q_u = A * F_R * [IAM * tau_alpha * G - U_L(v_wind) * (T_in - T_amb)] - Q_sky * F_R
    eta = Q_u / (A * G)

Note: For unglazed, tau_alpha ~ alpha (no glazing transmittance).

References:
    Duffie & Beckman (2013), 'Solar Engineering of Thermal Processes', Ch. 6, 10.
    ASHRAE Standard 93 (2010), Methods of Testing Solar Collectors.
    Martinopoulos et al. (2010), Solar Energy 84(1), 117-127.
    ISO 9806:2017, Solar energy — Solar thermal collectors — Test methods.
"""

import numpy as np


class UnglazedCollectorF1b:
    """Unglazed solar collector with IAM and wind-corrected heat loss."""

    SIGMA = 5.670374419e-8   # W/m2K4

    def __init__(self, params: dict):
        u = params["unit"]
        self.area = u["area"]["value"]              # m2
        self.F_R = u["F_R"]["value"]               # heat removal factor
        self.tau_alpha = u["tau_alpha"]["value"]    # absorptance (no glazing)
        self.U_L0 = u["U_L0"]["value"]             # W/m2K — base heat loss coeff
        self.U_wind = u["U_wind"]["value"]          # W/(m2K·m/s)
        self.b0 = u["b0"]["value"]                  # IAM coefficient
        self.eps_col = u["eps_col"]["value"]        # collector emissivity
        self.T_sky_offset = u["T_sky_offset"]["value"]  # K
        self.m_dot = u["m_dot"]["value"]            # kg/s
        self.cp = u["cp"]["value"]                  # J/kgK

    # ------------------------------------------------------------------
    # Incidence Angle Modifier (ASHRAE b0 model)
    # ------------------------------------------------------------------

    def iam(self, theta_deg):
        """
        IAM(theta) = 1 - b0*(1/cos(theta) - 1)
        ASHRAE 93 standard for unglazed collector surface.
        For polymer mat (smooth surface): b0 ~ 0.05-0.10 (much lower than glazed).
        """
        theta = np.asarray(theta_deg, dtype=float)
        theta_rad = np.radians(theta)
        cos_theta = np.cos(theta_rad)
        safe_cos = np.where(theta < 80.0, np.maximum(cos_theta, 0.01), 0.01)
        iam_val = 1.0 - self.b0 * (1.0 / safe_cos - 1.0)
        iam_val = np.where(theta < 80.0, iam_val, 0.0)
        return np.clip(iam_val, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Wind-corrected heat loss coefficient
    # ------------------------------------------------------------------

    def U_L(self, v_wind):
        """
        Overall heat loss coefficient corrected for wind speed.
        U_L(v) = U_L0 + U_wind * v_wind
        Typical for unglazed EPDM: U_L0 ~ 15 W/m2K, U_wind ~ 3 W/(m2K·m/s).
        """
        v = np.asarray(v_wind, dtype=float)
        return self.U_L0 + self.U_wind * v

    # ------------------------------------------------------------------
    # Sky radiation loss
    # ------------------------------------------------------------------

    def Q_sky_loss_w(self, T_col_c, T_ambient_c):
        """
        Net longwave radiation loss to cold sky (W).
        Q_sky = eps * sigma * A * (T_col^4 - T_sky^4)
        Unglazed collectors directly exposed to sky — this term is significant.
        """
        T_col = np.asarray(T_col_c, dtype=float) + 273.15
        T_amb = np.asarray(T_ambient_c, dtype=float) + 273.15
        T_sky = T_amb - self.T_sky_offset
        q_rad = self.eps_col * self.SIGMA * (T_col**4 - T_sky**4)  # W/m2
        return self.area * np.maximum(0.0, q_rad)

    # ------------------------------------------------------------------
    # Useful heat
    # ------------------------------------------------------------------

    def useful_heat_w(self, irradiance, incidence_angle_deg, T_inlet_c, T_ambient_c,
                      v_wind=2.0):
        """
        Useful heat gain [W].

        Q_u = A*F_R*[IAM*tau_alpha*G - U_L(v)*(T_in - T_amb)]
                  - F_R * Q_sky_loss
        """
        G = np.asarray(irradiance, dtype=float)
        T_in = np.asarray(T_inlet_c, dtype=float)
        T_amb = np.asarray(T_ambient_c, dtype=float)

        iam_val = self.iam(incidence_angle_deg)
        U_l = self.U_L(v_wind)

        # Approximate collector temperature for sky loss
        T_col_approx = T_in + 2.0  # assume small temperature rise, use inlet + offset

        Q_sky = self.Q_sky_loss_w(T_col_approx, T_ambient_c)

        Q_u = (self.area * self.F_R * (iam_val * self.tau_alpha * G
               - U_l * (T_in - T_amb))
               - self.F_R * Q_sky)
        return np.maximum(0.0, Q_u)

    # ------------------------------------------------------------------
    # Efficiency
    # ------------------------------------------------------------------

    def efficiency(self, irradiance, incidence_angle_deg, T_inlet_c, T_ambient_c,
                   v_wind=2.0):
        """Collector efficiency eta = Q_u / (A * G)."""
        G = np.asarray(irradiance, dtype=float)
        Q_u = self.useful_heat_w(irradiance, incidence_angle_deg, T_inlet_c,
                                  T_ambient_c, v_wind)
        denom = self.area * np.where(G > 1.0, G, 1.0)
        eta = np.where(G > 1.0, Q_u / denom, 0.0)
        return np.clip(eta, 0.0, self.tau_alpha)

    # ------------------------------------------------------------------
    # Outlet temperature
    # ------------------------------------------------------------------

    def T_outlet(self, irradiance, incidence_angle_deg, T_inlet_c, T_ambient_c,
                 v_wind=2.0):
        """Outlet temperature [degC]."""
        Q_u = self.useful_heat_w(irradiance, incidence_angle_deg, T_inlet_c,
                                  T_ambient_c, v_wind)
        T_in = np.asarray(T_inlet_c, dtype=float)
        return T_in + Q_u / (self.m_dot * self.cp)

    # ------------------------------------------------------------------
    # predict_all
    # ------------------------------------------------------------------

    def predict_all(self, irradiance, incidence_angle_deg, T_inlet_c, T_ambient_c,
                    v_wind=2.0):
        """
        Return all outputs as a dict.

        Parameters
        ----------
        irradiance         : W/m2
        incidence_angle_deg: degrees
        T_inlet_c          : degC — pool/fluid inlet temperature
        T_ambient_c        : degC
        v_wind             : m/s — wind speed (default 2 m/s)
        """
        Q_u   = self.useful_heat_w(irradiance, incidence_angle_deg, T_inlet_c,
                                    T_ambient_c, v_wind)
        eta   = self.efficiency(irradiance, incidence_angle_deg, T_inlet_c,
                                 T_ambient_c, v_wind)
        iam_v = self.iam(incidence_angle_deg)
        T_out = self.T_outlet(irradiance, incidence_angle_deg, T_inlet_c,
                               T_ambient_c, v_wind)
        U_l   = self.U_L(v_wind)
        Q_sky = self.Q_sky_loss_w(T_inlet_c, T_ambient_c)

        return {
            "useful_heat_w":   Q_u,
            "efficiency":      eta,
            "iam_factor":      iam_v,
            "T_outlet_degC":   T_out,
            "U_L_effective":   np.asarray(U_l, dtype=float),
            "Q_sky_loss_w":    Q_sky,
        }
