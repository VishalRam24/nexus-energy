"""
EC060 — Solar Pond — F1b Incidence Angle Modifier Model

A solar pond stores solar energy as heat in a saline water body. The non-convective
salt-gradient zone (NCZ) suppresses natural convection, allowing the lower convective
zone (LCZ) to reach 70–90 °C.

Extends F1a (Hottel-Whillier for pond) by adding:
  1. Refraction-corrected IAM for water surface (Snell's law)
  2. Wavelength-integrated transmittance through brine layers
  3. Separate extraction efficiency for the lower convective zone (LCZ)

Hottel-Whillier for solar pond (LCZ):
    Q_u = A * [tau_pond * IAM(theta) * G * alpha_lcz - U_lcz * (T_lcz - T_amb)]
    eta = Q_u / (A * G)

IAM for water surface (Snell-law refraction):
    theta_r = arcsin(sin(theta) / n_brine)    [refraction into brine]
    IAM(theta) = tau(theta) / tau(0)
    Using Fresnel reflectance approximation:
        rho_s = [(cos(theta) - n*cos(theta_r)) / (cos(theta) + n*cos(theta_r))]^2
        rho_p = [(n*cos(theta) - cos(theta_r)) / (n*cos(theta) + cos(theta_r))]^2
        tau = 1 - (rho_s + rho_p) / 2
    IAM = tau(theta) / tau(0)

References:
    Duffie & Beckman (2013), 'Solar Engineering of Thermal Processes', Ch. 9.
    Tabor & Matz (1965), Solar Pond project. Solar Energy 9(4), 177-182.
    Singh et al. (2011), Review of solar pond technology. Renewable and Sustainable
    Energy Reviews 15(4), 1773-1781.
"""

import numpy as np


class SolarPondF1b:
    """Solar pond — Hottel-Whillier model with Snell-law IAM for brine surface."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.area = u["area"]["value"]              # m2 — pond surface area
        self.tau_pond = u["tau_pond"]["value"]       # transmittance product through brine (normal inc.)
        self.alpha_lcz = u["alpha_lcz"]["value"]     # absorptance of LCZ (dark bottom)
        self.U_lcz = u["U_lcz"]["value"]             # W/m2K — overall loss from LCZ
        self.n_brine = u["n_brine"]["value"]         # refractive index of brine (~1.40)
        self.m_dot = u["m_dot"]["value"]             # kg/s — extraction flow rate
        self.cp_brine = u["cp_brine"]["value"]       # J/kgK

    # ------------------------------------------------------------------
    # Fresnel IAM for water/brine surface
    # ------------------------------------------------------------------

    def iam(self, theta_deg):
        """
        Incidence Angle Modifier using Fresnel reflectance and Snell's law.
        IAM(theta) = tau(theta) / tau(0)

        At normal incidence (theta=0), tau(0) is computed from Fresnel (it is
        close to 1 for water, as reflectance ~ 0.02).
        """
        theta = np.asarray(theta_deg, dtype=float)
        theta_rad = np.radians(np.minimum(theta, 89.0))

        # Snell's law refraction angle
        sin_theta_r = np.sin(theta_rad) / self.n_brine
        sin_theta_r = np.minimum(sin_theta_r, 1.0 - 1e-9)
        theta_r_rad = np.arcsin(sin_theta_r)

        cos_i = np.cos(theta_rad)
        cos_r = np.cos(theta_r_rad)
        n = self.n_brine

        # Fresnel s- and p-polarisation reflectances
        denom_s = cos_i + n * cos_r
        denom_p = n * cos_i + cos_r
        rho_s = np.where(np.abs(denom_s) > 1e-12,
                         ((cos_i - n * cos_r) / denom_s) ** 2, 1.0)
        rho_p = np.where(np.abs(denom_p) > 1e-12,
                         ((n * cos_i - cos_r) / denom_p) ** 2, 1.0)
        tau_theta = 1.0 - (rho_s + rho_p) / 2.0

        # tau at normal incidence (theta=0)
        # Fresnel at 0: rho_s = rho_p = ((1-n)/(1+n))^2
        rho0 = ((1.0 - n) / (1.0 + n)) ** 2
        tau_0 = 1.0 - rho0

        iam_val = np.where(tau_0 > 1e-10, tau_theta / tau_0, 0.0)
        iam_val = np.where(theta < 89.0, iam_val, 0.0)
        return np.clip(iam_val, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Useful heat from LCZ
    # ------------------------------------------------------------------

    def useful_heat_w(self, irradiance, incidence_angle_deg, T_lcz_c, T_ambient_c):
        """
        Useful heat from the Lower Convective Zone [W].

        Q_u = A * [tau_pond * IAM * alpha_lcz * G - U_lcz * (T_lcz - T_amb)]
        """
        G = np.asarray(irradiance, dtype=float)
        T_lcz = np.asarray(T_lcz_c, dtype=float)
        T_amb = np.asarray(T_ambient_c, dtype=float)

        iam_val = self.iam(incidence_angle_deg)

        Q_u = self.area * (self.tau_pond * iam_val * self.alpha_lcz * G
                           - self.U_lcz * (T_lcz - T_amb))
        return np.maximum(0.0, Q_u)

    # ------------------------------------------------------------------
    # Efficiency
    # ------------------------------------------------------------------

    def efficiency(self, irradiance, incidence_angle_deg, T_lcz_c, T_ambient_c):
        """Instantaneous thermal efficiency eta = Q_u / (A * G)."""
        G = np.asarray(irradiance, dtype=float)
        Q_u = self.useful_heat_w(irradiance, incidence_angle_deg, T_lcz_c, T_ambient_c)
        denom = self.area * np.where(G > 1.0, G, 1.0)
        eta = np.where(G > 1.0, Q_u / denom, 0.0)
        return np.clip(eta, 0.0, self.tau_pond * self.alpha_lcz)

    # ------------------------------------------------------------------
    # Outlet temperature from extraction heat exchanger
    # ------------------------------------------------------------------

    def T_outlet_extraction(self, irradiance, incidence_angle_deg, T_lcz_c, T_ambient_c):
        """
        Temperature rise of extraction fluid from LCZ.
        T_out = T_in_extraction + Q_u / (m_dot * cp)
        where T_in_extraction ~ T_lcz (drawn from bottom).
        """
        Q_u = self.useful_heat_w(irradiance, incidence_angle_deg, T_lcz_c, T_ambient_c)
        T_lcz = np.asarray(T_lcz_c, dtype=float)
        # Extraction fluid enters at near-LCZ temperature
        T_out = T_lcz + Q_u / (self.m_dot * self.cp_brine)
        return T_out

    # ------------------------------------------------------------------
    # predict_all
    # ------------------------------------------------------------------

    def predict_all(self, irradiance, incidence_angle_deg, T_lcz_c, T_ambient_c):
        """
        Return all outputs as a dict.

        Parameters
        ----------
        irradiance         : global horizontal irradiance (W/m2)
        incidence_angle_deg: solar zenith angle (deg) — equivalent for horizontal pond
        T_lcz_c            : LCZ temperature (degC)
        T_ambient_c        : ambient/UCZ temperature (degC)
        """
        Q_u   = self.useful_heat_w(irradiance, incidence_angle_deg, T_lcz_c, T_ambient_c)
        eta   = self.efficiency(irradiance, incidence_angle_deg, T_lcz_c, T_ambient_c)
        iam_v = self.iam(incidence_angle_deg)
        T_out = self.T_outlet_extraction(irradiance, incidence_angle_deg, T_lcz_c, T_ambient_c)

        return {
            "useful_heat_w":    Q_u,
            "efficiency":       eta,
            "iam_factor":       iam_v,
            "T_extraction_degC": T_out,
        }
