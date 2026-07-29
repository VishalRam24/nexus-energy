"""
EC054 — Parabolic Trough CSP — F1a Optical Efficiency + Heat Loss Model

Q_useful = DNI * A_aperture * eta_optical * IAM(theta) - Q_loss_total

Optical:
    eta_optical = reflectivity * intercept_factor * transmissivity * absorptivity
    IAM(theta)  = 1 - IAM_coeff * theta^2   (Incidence Angle Modifier)

Receiver heat loss (Schott PTR70, per metre of receiver tube):
    q_loss = a0 + a1*(T_abs - T_amb) + a2*(T_abs - T_amb)^2   [W/m]
    Q_loss_total = q_loss * L_collector   [W -> kW]

Overall efficiency:
    eta_overall = Q_useful / (DNI * A_aperture)   when DNI > 0

References:
    Forristall (2003), 'Heat Transfer Analysis and Modeling of a Parabolic Trough
    Solar Receiver Implemented in Engineering Equation Solver',
    NREL/TP-550-34169, National Renewable Energy Laboratory.
"""

import numpy as np


class ParabolicTroughF1a:
    """Parabolic trough CSP — optical efficiency and receiver heat loss model."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.A_aperture = u["A_aperture"]["value"]    # m2
        self.eta_optical = u["eta_optical"]["value"]   # peak optical efficiency
        self.IAM_coeff = u["IAM_coeff"]["value"]       # 1/deg2
        self.a0 = u["a0"]["value"]                    # W/m
        self.a1 = u["a1"]["value"]                    # W/(m*K)
        self.a2 = u["a2"]["value"]                    # W/(m*K2)
        self.L = u["L_collector"]["value"]             # m

    def IAM(self, theta_deg):
        """Incidence Angle Modifier. IAM=1 at normal incidence (theta=0)."""
        theta = np.asarray(theta_deg, dtype=float)
        iam = 1.0 - self.IAM_coeff * theta ** 2
        return np.clip(iam, 0.0, 1.0)

    def heat_loss_kw(self, T_absorber_c, T_ambient_c):
        """Total receiver heat loss in kW."""
        dT = np.asarray(T_absorber_c, dtype=float) - np.asarray(T_ambient_c, dtype=float)
        q_loss_per_m = self.a0 + self.a1 * dT + self.a2 * dT ** 2  # W/m
        return q_loss_per_m * self.L / 1000.0  # kW

    def Q_absorbed_kw(self, dni, theta_deg):
        """Solar power absorbed by receiver (before heat loss), kW."""
        G = np.asarray(dni, dtype=float)
        iam = self.IAM(theta_deg)
        return G * self.A_aperture * self.eta_optical * iam / 1000.0  # kW

    def predict_all(self, dni, T_absorber_c, T_ambient_c, incidence_angle_deg):
        """
        Compute all outputs.

        Parameters
        ----------
        dni                : Direct Normal Irradiance (W/m2)
        T_absorber_c       : HTF / absorber tube temperature (degC)
        T_ambient_c        : ambient temperature (degC)
        incidence_angle_deg: angle between sun vector and collector normal (deg)

        Returns
        -------
        dict: useful_heat_kw, optical_efficiency, thermal_loss_kw, overall_efficiency
        """
        Q_abs = self.Q_absorbed_kw(dni, incidence_angle_deg)
        Q_loss = self.heat_loss_kw(T_absorber_c, T_ambient_c)
        Q_useful = np.maximum(0.0, Q_abs - Q_loss)

        G = np.asarray(dni, dtype=float)
        P_incident_safe = np.where(G > 0.01, G * self.A_aperture / 1000.0, 1.0)  # kW, safe denom
        eta_opt = np.where(G > 0.01, Q_abs / P_incident_safe, 0.0)
        eta_overall = np.where(G > 0.01, Q_useful / P_incident_safe, 0.0)

        return {
            "useful_heat_kw": Q_useful,
            "optical_efficiency": np.clip(eta_opt, 0.0, 1.0),
            "thermal_loss_kw": Q_loss,
            "overall_efficiency": np.clip(eta_overall, 0.0, 1.0),
        }
