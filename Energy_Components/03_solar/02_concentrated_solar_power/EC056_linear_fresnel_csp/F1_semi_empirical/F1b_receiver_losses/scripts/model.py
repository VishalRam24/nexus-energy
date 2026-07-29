"""
EC056 — Linear Fresnel CSP — F1b Receiver Heat Loss Model

Extends F1a (optical efficiency) with detailed receiver thermal losses:
    Q_loss = pi*D_abs*L * [h_conv*(T_abs - T_amb) + eps*sigma*(T_abs^4 - T_sky^4)]

Linear Fresnel vs Parabolic Trough differences (EC054):
  - Multiple flat or slightly curved mirror rows reflect to a single elevated absorber
  - Trapezoidal cavity secondary reflector concentrates onto absorber
  - End effects: no tracking in one axis → cosine loss is more significant than trough
  - IAM is product of longitudinal IAM and transversal IAM
  - Concentrator ratio ~20-80x (lower than parabolic trough ~80-100x)

IAM for Linear Fresnel (two-axis decomposition per Incropera (2011) / IRENA 2012):
    IAM(theta_L, theta_T) = IAM_L(theta_L) * IAM_T(theta_T)
    IAM_L(theta_L) = cos(theta_L)  (longitudinal angle: along absorber axis)
    IAM_T(theta_T) = 1 - b_T * theta_T^2  (transversal: across absorber, polynomial)

End loss:
    f_end = 1 - f_L * tan(theta_L) / L_collector  (same form as parabolic trough)

References:
    Montes et al. (2009). "Performance analysis of an integrated solar combined cycle
    using Direct Steam Generation in parabolic troughs." Appl. Energy 86, 2080-2092.
    Zhu et al. (2014). "History, current state, and future of linear Fresnel concentrating
    solar collectors." Solar Energy 103, 639-652.
    Häberle et al. (2002). "The Solarmundo line focussing Fresnel collector."
    Eurosun 2002 Proceedings.
    Forristall (2003). NREL/TP-550-34169.
"""

import numpy as np


class LinearFresnelF1b:
    """Linear Fresnel CSP — detailed receiver heat loss + two-axis IAM model."""

    SIGMA = 5.670374419e-8  # Stefan-Boltzmann [W/m2K4]

    def __init__(self, params: dict):
        u = params["unit"]
        iam = params["iam_coefficients"]

        self.W_total = u["total_mirror_width"]["value"]     # m (sum of all mirror row widths)
        self.L = u["L_collector"]["value"]                  # m
        self.D_abs = u["D_abs"]["value"]                    # m
        self.eps_abs = u["eps_abs"]["value"]
        self.h_conv = u["h_conv"]["value"]                  # W/m2K (cavity convection)
        self.T_sky_offset = u["T_sky_offset"]["value"]      # K
        self.focal_length = u["focal_length"]["value"]      # m (height of absorber)

        self.rho_mirror = u["rho_mirror"]["value"]
        self.intercept_factor = u["intercept_factor"]["value"]
        self.tau_secondary = u["tau_secondary"]["value"]    # secondary reflector transmittance
        self.abs_absorptance = u["abs_absorptance"]["value"]  # absorber absorptance

        # IAM coefficients
        self.b_T = iam["b_T"]["value"]           # transversal polynomial coefficient

        self.A_aperture = self.W_total * self.L

    def iam_longitudinal(self, theta_L_deg):
        """
        Longitudinal IAM = cos(theta_L).
        For linear Fresnel, row-mirror focus is maintained in longitudinal direction.
        """
        theta_L = np.asarray(theta_L_deg, dtype=float)
        theta_L_rad = np.radians(theta_L)
        return np.clip(np.cos(theta_L_rad), 0.0, 1.0)

    def iam_transversal(self, theta_T_deg):
        """
        Transversal IAM (polynomial fit to Fresnel mirror row shading/blocking).
        IAM_T = 1 - b_T * theta_T^2
        Based on Häberle et al. (2002) for Solarmundo-type geometry.
        Returns 0 for theta_T > 60 deg (beyond collector acceptance).
        """
        theta_T = np.asarray(theta_T_deg, dtype=float)
        iam_val = 1.0 - self.b_T * theta_T**2
        iam_val = np.where(theta_T < 60.0, iam_val, 0.0)
        return np.clip(iam_val, 0.0, 1.0)

    def end_loss_factor(self, theta_L_deg):
        """End loss factor for finite collector length."""
        theta_L = np.asarray(theta_L_deg, dtype=float)
        theta_L_rad = np.radians(theta_L)
        tan_theta = np.where(theta_L < 85.0, np.tan(theta_L_rad), 100.0)
        f_end = 1.0 - self.focal_length * tan_theta / self.L
        return np.clip(f_end, 0.0, 1.0)

    def optical_efficiency(self, theta_L_deg, theta_T_deg):
        """
        Total optical efficiency.
        eta_opt = rho * intercept * tau_sec * alpha_abs * IAM_L * IAM_T * f_end
        """
        iam_L = self.iam_longitudinal(theta_L_deg)
        iam_T = self.iam_transversal(theta_T_deg)
        f_end = self.end_loss_factor(theta_L_deg)
        return (self.rho_mirror * self.intercept_factor * self.tau_secondary
                * self.abs_absorptance * iam_L * iam_T * f_end)

    def receiver_loss_kw_per_m(self, T_abs_c, T_amb_c):
        """
        Receiver heat loss per metre of absorber tube.
        Q_loss/L = pi*D_abs * [h_conv*(T_abs - T_amb) + eps*sigma*(T_abs^4 - T_sky^4)]

        Cavity receiver reduces convective loss coefficient compared to exposed
        parabolic trough receiver.
        """
        T_abs = np.asarray(T_abs_c, dtype=float) + 273.15
        T_amb = np.asarray(T_amb_c, dtype=float) + 273.15
        T_sky = T_amb - self.T_sky_offset

        q_conv = self.h_conv * (T_abs - T_amb)
        q_rad = self.eps_abs * self.SIGMA * (T_abs**4 - T_sky**4)
        q_loss_per_m = np.pi * self.D_abs * (q_conv + q_rad)
        return q_loss_per_m / 1000.0  # kW/m

    def predict_all(self, dni, T_htf_in_c, T_htf_out_c, T_ambient_c,
                    theta_L_deg, theta_T_deg):
        """
        Compute all outputs for linear Fresnel collector.

        Parameters
        ----------
        dni          : W/m2, Direct Normal Irradiance
        T_htf_in_c   : degC, HTF inlet temperature
        T_htf_out_c  : degC, HTF outlet temperature
        T_ambient_c  : degC, ambient temperature
        theta_L_deg  : degrees, longitudinal incidence angle (along collector axis)
        theta_T_deg  : degrees, transversal incidence angle (across collector axis)
        """
        G = np.asarray(dni, dtype=float)
        T_abs = (np.asarray(T_htf_in_c, dtype=float) + np.asarray(T_htf_out_c, dtype=float)) / 2.0

        eta_opt = self.optical_efficiency(theta_L_deg, theta_T_deg)
        q_solar_per_m = G * self.W_total * eta_opt / 1000.0  # kW/m

        q_loss_per_m = self.receiver_loss_kw_per_m(T_abs, T_ambient_c)
        q_useful_per_m = np.maximum(0.0, q_solar_per_m - q_loss_per_m)

        q_incident_per_m = G * self.W_total / 1000.0
        safe_incident = np.where(q_incident_per_m > 0.001, q_incident_per_m, 1.0)
        eta_thermal = np.where(q_incident_per_m > 0.001,
                               q_useful_per_m / safe_incident, 0.0)

        return {
            "thermal_output_kw_per_m": q_useful_per_m,
            "optical_efficiency": np.clip(eta_opt, 0.0, 1.0),
            "thermal_efficiency": np.clip(eta_thermal, 0.0, 1.0),
            "receiver_loss_kw_per_m": q_loss_per_m,
            "iam_longitudinal": self.iam_longitudinal(theta_L_deg),
            "iam_transversal": self.iam_transversal(theta_T_deg),
        }
