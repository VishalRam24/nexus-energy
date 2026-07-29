"""
EC054 — Parabolic Trough CSP — F1b Receiver Heat Loss Model

Detailed receiver loss model:
    Q_loss = pi*D_abs*L * [h_conv*(T_abs - T_amb) + eps*sigma*(T_abs^4 - T_sky^4)]

Polynomial Incidence Angle Modifier:
    IAM(theta) = cos(theta) - 5.25097e-4*theta - 2.859621e-5*theta^2

End loss factor:
    f_end = 1 - f_L*tan(theta) / L_collector
    where f_L = focal_length

Improvement over F1a:
    - Physics-based convective + radiative loss instead of polynomial fit
    - Polynomial IAM from Sandia testing (more accurate than quadratic)
    - End loss correction for finite collector length

References:
    Forristall (2003), 'Heat Transfer Analysis and Modeling of a Parabolic Trough
    Solar Receiver', NREL/TP-550-34169.
    Dudley et al. (1994), 'Test Results: SEGS LS-2 Solar Collector', SAND94-1884.
    Kalogirou (2012), Solar Energy 86(1), 1-17.
"""

import numpy as np


class ParabolicTroughF1b:
    """Parabolic trough CSP — detailed receiver heat loss model."""

    SIGMA = 5.670374419e-8  # Stefan-Boltzmann constant [W/m2K4]

    def __init__(self, params: dict):
        u = params["unit"]
        iam = params["iam_coefficients"]

        self.W = u["aperture_width"]["value"]       # m
        self.L = u["L_collector"]["value"]           # m
        self.D_abs = u["D_abs"]["value"]             # m
        self.eps_abs = u["eps_abs"]["value"]
        self.eps_glass = u["eps_glass"]["value"]
        self.tau_glass = u["tau_glass"]["value"]
        self.rho_mirror = u["rho_mirror"]["value"]
        self.intercept_factor = u["intercept_factor"]["value"]
        self.h_conv = u["h_conv"]["value"]           # W/m2K
        self.T_sky_offset = u["T_sky_offset"]["value"]  # K
        self.focal_length = u["focal_length"]["value"]  # m

        # IAM polynomial coefficients
        self.iam_c1 = iam["c1"]["value"]
        self.iam_c2 = iam["c2"]["value"]

        # Derived
        self.A_aperture = self.W * self.L           # m2

    def iam(self, theta_deg):
        """
        Incidence Angle Modifier (polynomial, Dudley et al. 1994).
        IAM(theta) = cos(theta) - c1*theta - c2*theta^2
        """
        theta = np.asarray(theta_deg, dtype=float)
        theta_rad = np.radians(theta)
        iam_val = np.cos(theta_rad) + self.iam_c1 * theta + self.iam_c2 * theta**2
        return np.clip(iam_val, 0.0, 1.0)

    def end_loss_factor(self, theta_deg):
        """
        End loss factor for finite collector length.
        f_end = 1 - f_L * tan(theta) / L
        """
        theta = np.asarray(theta_deg, dtype=float)
        theta_rad = np.radians(theta)
        # Avoid tan(90)
        tan_theta = np.where(theta < 85.0, np.tan(theta_rad), 100.0)
        f_end = 1.0 - self.focal_length * tan_theta / self.L
        return np.clip(f_end, 0.0, 1.0)

    def optical_efficiency(self, theta_deg):
        """
        Total optical efficiency including IAM, end loss, mirror, glass, intercept.
        eta_opt = rho * intercept * tau_glass * eps_abs * IAM * f_end
        """
        iam_val = self.iam(theta_deg)
        f_end = self.end_loss_factor(theta_deg)
        return self.rho_mirror * self.intercept_factor * self.tau_glass * self.eps_abs * iam_val * f_end

    def receiver_loss_kw_per_m(self, T_abs_c, T_amb_c):
        """
        Receiver heat loss per metre of absorber tube (kW/m).

        Q_loss/L = pi*D_abs * [h_conv*(T_abs - T_amb) + eps*sigma*(T_abs^4 - T_sky^4)]
        """
        T_abs = np.asarray(T_abs_c, dtype=float) + 273.15   # K
        T_amb = np.asarray(T_amb_c, dtype=float) + 273.15   # K
        T_sky = T_amb - self.T_sky_offset                     # K

        # Convective loss (external, annulus approximation)
        q_conv = self.h_conv * (T_abs - T_amb)

        # Radiative loss (absorber to sky through glass)
        q_rad = self.eps_abs * self.SIGMA * (T_abs**4 - T_sky**4)

        q_loss_per_m = np.pi * self.D_abs * (q_conv + q_rad)  # W/m
        return q_loss_per_m / 1000.0  # kW/m

    def predict_all(self, dni, T_htf_in_c, T_htf_out_c, T_ambient_c, incidence_angle_deg):
        """
        Compute all outputs.

        Parameters
        ----------
        dni                 : Direct Normal Irradiance (W/m2)
        T_htf_in_c          : HTF inlet temperature (degC)
        T_htf_out_c         : HTF outlet temperature (degC)
        T_ambient_c         : Ambient temperature (degC)
        incidence_angle_deg : Sun incidence angle (deg)

        Returns
        -------
        dict: thermal_output_kw_per_m, optical_efficiency, thermal_efficiency,
              receiver_loss_kw_per_m
        """
        G = np.asarray(dni, dtype=float)
        T_in = np.asarray(T_htf_in_c, dtype=float)
        T_out = np.asarray(T_htf_out_c, dtype=float)
        theta = np.asarray(incidence_angle_deg, dtype=float)

        # Average absorber temperature
        T_abs = (T_in + T_out) / 2.0

        # Optical power collected per unit length (kW/m)
        eta_opt = self.optical_efficiency(theta)
        q_solar_per_m = G * self.W * eta_opt / 1000.0  # kW/m

        # Receiver thermal losses per unit length
        q_loss_per_m = self.receiver_loss_kw_per_m(T_abs, T_ambient_c)

        # Net thermal output per unit length
        q_useful_per_m = np.maximum(0.0, q_solar_per_m - q_loss_per_m)

        # Thermal efficiency = q_useful / q_solar_incident
        q_incident_per_m = G * self.W / 1000.0  # kW/m
        safe_incident = np.where(q_incident_per_m > 0.001, q_incident_per_m, 1.0)
        eta_thermal = np.where(q_incident_per_m > 0.001,
                               q_useful_per_m / safe_incident, 0.0)

        return {
            "thermal_output_kw_per_m": q_useful_per_m,
            "optical_efficiency": np.clip(eta_opt, 0.0, 1.0),
            "thermal_efficiency": np.clip(eta_thermal, 0.0, 1.0),
            "receiver_loss_kw_per_m": q_loss_per_m,
        }
