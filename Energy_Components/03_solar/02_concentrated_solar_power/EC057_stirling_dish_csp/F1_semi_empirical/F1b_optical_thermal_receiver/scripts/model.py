"""
EC057 — Stirling Dish CSP — F1b Optical + Receiver Thermal Loss Model

Extends F1a (optical-only) by adding:
  1. Physics-based receiver (cavity) heat loss: convective + radiative
  2. Stirling engine cycle efficiency corrected for hot-end temperature
  3. Part-load correction via PLR curve

Dish optical model:
    Q_absorbed = DNI * A_dish * eta_optical * IAM(theta)
    IAM(theta) = cos(theta)  (dish tracks to minimise incidence)

Receiver (cavity) losses:
    Q_conv = h_cav * A_rec * (T_rec - T_amb)
    Q_rad  = eps_rec * sigma * A_rec * (T_rec^4 - T_sky^4)
    Q_cond = U_cond * A_rec * (T_rec - T_amb)
    Q_loss = Q_conv + Q_rad + Q_cond

Stirling engine thermal efficiency (Schmidt/first-law approximation):
    eta_stirling = eta_internal * (1 - T_sink / T_hot)   [modified Carnot]
    where T_hot = T_rec - dT_receiver (fluid side after receiver losses)
    and T_sink = T_amb + T_approach

Part-load correction:
    f_PLR(PLR) = plr_a + plr_b * PLR + plr_c * PLR^2

Net electrical output:
    P_elec = (Q_absorbed - Q_loss) * eta_stirling * f_PLR * eta_alt

References:
    Stine & Diver (1994), 'A Compendium of Solar Dish/Stirling Technology',
    SAND93-7026, Sandia National Laboratories.
    Mancini et al. (2003), 'Dish-Stirling Systems: An Overview of Development
    and Status', J. Sol. Energy Eng. 125(2), 135-151.
    Nepveu et al. (2009), 'Thermal model of a dish/Stirling systems',
    Sol. Energy 83(1), 81-89.
"""

import numpy as np


class StirlingDishF1b:
    """Stirling Dish CSP — optical efficiency + receiver thermal loss model."""

    SIGMA = 5.670374419e-8  # W/m2K4

    def __init__(self, params: dict):
        u = params["unit"]
        self.A_dish = u["A_dish"]["value"]           # m2  — dish aperture area
        self.eta_optical = u["eta_optical"]["value"] # dimensionless — peak optical
        self.A_rec = u["A_rec"]["value"]             # m2  — receiver aperture area
        self.eps_rec = u["eps_rec"]["value"]         # emissivity of receiver cavity
        self.h_cav = u["h_cav"]["value"]             # W/m2K — natural convection
        self.U_cond = u["U_cond"]["value"]           # W/m2K — conduction loss
        self.T_sky_offset = u["T_sky_offset"]["value"]  # K offset for sky temp
        self.T_rec_design = u["T_rec_design"]["value"]   # degC — design receiver temp
        self.dT_receiver = u["dT_receiver"]["value"]     # K — receiver to Stirling hot end
        self.T_approach = u["T_approach"]["value"]        # K — Stirling sink approach
        self.eta_internal = u["eta_internal"]["value"]   # Stirling internal efficiency
        self.eta_alt = u["eta_alt"]["value"]              # alternator efficiency
        self.plr_a = u["plr_a"]["value"]
        self.plr_b = u["plr_b"]["value"]
        self.plr_c = u["plr_c"]["value"]
        self.PLR_min = u["PLR_min"]["value"]
        self.P_rated = u["P_rated_kw"]["value"]          # kW

    # ------------------------------------------------------------------
    # Incidence Angle Modifier (dish tracks sun — IAM = cos(theta))
    # ------------------------------------------------------------------

    def iam(self, theta_deg):
        """Incidence Angle Modifier for dish — single-axis tracking.
        Residual incidence angle theta arises from tracking error only.
        IAM = cos(theta), valid for theta < 80 deg.
        """
        theta = np.asarray(theta_deg, dtype=float)
        theta_rad = np.radians(np.minimum(theta, 89.9))
        iam_val = np.cos(theta_rad)
        return np.where(theta < 80.0, np.clip(iam_val, 0.0, 1.0), 0.0)

    # ------------------------------------------------------------------
    # Solar power absorbed
    # ------------------------------------------------------------------

    def Q_absorbed_kw(self, dni, theta_deg):
        """Solar power delivered to receiver aperture [kW]."""
        G = np.asarray(dni, dtype=float)
        iam_val = self.iam(theta_deg)
        return G * self.A_dish * self.eta_optical * iam_val / 1000.0  # kW

    # ------------------------------------------------------------------
    # Receiver thermal losses
    # ------------------------------------------------------------------

    def Q_receiver_loss_kw(self, T_rec_c, T_amb_c):
        """
        Total receiver cavity heat loss [kW].
        Q_loss = A_rec * [h_cav*(T_rec - T_amb) + U_cond*(T_rec - T_amb)
                          + eps*sigma*(T_rec^4 - T_sky^4)]
        """
        T_rec = np.asarray(T_rec_c, dtype=float) + 273.15   # K
        T_amb = np.asarray(T_amb_c, dtype=float) + 273.15   # K
        T_sky = T_amb - self.T_sky_offset                    # K

        q_conv = self.h_cav * (T_rec - T_amb)               # W/m2
        q_cond = self.U_cond * (T_rec - T_amb)              # W/m2
        q_rad  = self.eps_rec * self.SIGMA * (T_rec**4 - T_sky**4)  # W/m2

        q_total = self.A_rec * (q_conv + q_cond + q_rad)    # W
        return np.maximum(0.0, q_total / 1000.0)            # kW

    # ------------------------------------------------------------------
    # Stirling engine efficiency
    # ------------------------------------------------------------------

    def eta_stirling(self, T_rec_c, T_amb_c, PLR=1.0):
        """
        Net Stirling thermal-to-electrical efficiency.

        eta_stir = eta_internal * (1 - T_sink / T_hot) * f_PLR
        where T_hot = T_rec - dT_receiver [K]
              T_sink = T_amb + T_approach [K]
        """
        T_rec = np.asarray(T_rec_c, dtype=float)
        T_hot = T_rec - self.dT_receiver + 273.15           # K
        T_sink = np.asarray(T_amb_c, dtype=float) + self.T_approach + 273.15  # K

        dT = T_hot - T_sink
        eta_carnot = np.where(dT > 0.0, 1.0 - T_sink / T_hot, 0.0)

        PLR = np.asarray(PLR, dtype=float)
        PLR_eff = np.maximum(PLR, self.PLR_min)
        f_plr = self.plr_a + self.plr_b * PLR_eff + self.plr_c * PLR_eff**2

        eta = self.eta_internal * eta_carnot * f_plr * self.eta_alt
        # Must not exceed Carnot
        return np.clip(eta, 0.0, np.maximum(eta_carnot * self.eta_alt, 1e-6))

    # ------------------------------------------------------------------
    # Net electrical output
    # ------------------------------------------------------------------

    def power_output_kw(self, dni, theta_deg, T_rec_c, T_amb_c, PLR=1.0):
        """
        Net electrical output [kW].
        P = (Q_absorbed - Q_loss) * eta_Stirling
        """
        Q_abs = self.Q_absorbed_kw(dni, theta_deg)
        Q_loss = self.Q_receiver_loss_kw(T_rec_c, T_amb_c)
        Q_net = np.maximum(0.0, Q_abs - Q_loss)             # kW thermal to engine

        eta = self.eta_stirling(T_rec_c, T_amb_c, PLR)
        return Q_net * eta

    # ------------------------------------------------------------------
    # Overall efficiency
    # ------------------------------------------------------------------

    def overall_efficiency(self, dni, theta_deg, T_rec_c, T_amb_c, PLR=1.0):
        """System efficiency = P_elec / (DNI * A_dish)."""
        G = np.asarray(dni, dtype=float)
        P = self.power_output_kw(dni, theta_deg, T_rec_c, T_amb_c, PLR)
        P_incident = G * self.A_dish / 1000.0  # kW
        safe_inc = np.where(P_incident > 0.001, P_incident, 1.0)
        eta = np.where(P_incident > 0.001, P / safe_inc, 0.0)
        return np.clip(eta, 0.0, 0.40)

    # ------------------------------------------------------------------
    # predict_all
    # ------------------------------------------------------------------

    def predict_all(self, dni, theta_deg, T_rec_c, T_amb_c, PLR=1.0):
        """
        Return all outputs as a dict.

        Parameters
        ----------
        dni        : Direct Normal Irradiance [W/m2]
        theta_deg  : tracking residual incidence angle [deg]
        T_rec_c    : receiver cavity temperature [degC]
        T_amb_c    : ambient temperature [degC]
        PLR        : part-load ratio [0.3-1.0]
        """
        Q_abs  = self.Q_absorbed_kw(dni, theta_deg)
        Q_loss = self.Q_receiver_loss_kw(T_rec_c, T_amb_c)
        Q_net  = np.maximum(0.0, Q_abs - Q_loss)
        eta_s  = self.eta_stirling(T_rec_c, T_amb_c, PLR)
        P_elec = Q_net * eta_s
        eta_sys = self.overall_efficiency(dni, theta_deg, T_rec_c, T_amb_c, PLR)
        iam_val = self.iam(theta_deg)

        return {
            "power_output_kw":      P_elec,
            "Q_absorbed_kw":        Q_abs,
            "Q_receiver_loss_kw":   Q_loss,
            "Q_net_thermal_kw":     Q_net,
            "eta_stirling":         eta_s,
            "overall_efficiency":   eta_sys,
            "iam_factor":           iam_val,
        }
