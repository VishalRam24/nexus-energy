"""
EC059 — Evacuated Tube Solar Collector — F1b Thermal Loss Curve

Extends F1a (constant F_R*U_L Hottel-Whillier-Bliss) with:
  1. Second-order heat loss: U_L(DeltaT) = a1 + a2 * DeltaT
     (ISO 9806:2017 second-order efficiency equation)
  2. Incidence Angle Modifier (IAM): IAM(theta) = 1 - b0*(1/cos(theta) - 1)

ISO 9806:2017 second-order efficiency equation:
    eta = eta_0 * IAM(theta) - a1 * (T_m - T_amb) / G - a2 * (T_m - T_amb)^2 / G

where T_m = (T_in + T_out) / 2 is the mean fluid temperature.

Iterative solution because T_out depends on Q_u and T_m depends on T_out.

ETC vacuum insulation physics:
    - Vacuum suppresses conductive and convective heat losses between absorber
      and outer glass tube → very low a1 (~1.4 W/m2K vs ~4 for flat plate)
    - Residual second-order loss from end-cap conduction and residual gas
      gives small a2 (~0.012 W/m2K2 vs ~0.06 for flat plate)
    - This makes ETCs attractive for high DeltaT applications (DHW, industrial
      process heat, space heating in cold climates)

References:
    Duffie & Beckman (2013). 'Solar Engineering of Thermal Processes', 4th ed., Ch.6.
    ISO 9806:2017. 'Solar energy — Solar thermal collectors — Test methods.'
    SRCC OG-100 certified ETC ratings (www.solar-rating.org).
"""

import numpy as np


class EvacuatedTubeF1b:
    """Evacuated tube collector — ISO 9806 second-order efficiency + IAM."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.area = u["area"]["value"]
        self.eta_0 = u["eta_0"]["value"]
        self.a1 = u["a1"]["value"]
        self.a2 = u["a2"]["value"]
        self.b0 = u["b0"]["value"]
        self.m_dot = u["m_dot"]["value"]
        self.cp = u["cp_fluid"]["value"]

    # ------------------------------------------------------------------
    # Incidence Angle Modifier
    # ------------------------------------------------------------------
    def iam(self, theta_deg):
        """
        Incidence Angle Modifier (Duffie & Beckman Eq. 6.17.6).
        IAM(theta) = 1 - b0 * (1/cos(theta) - 1)
        Returns 0 for theta >= 80 deg.
        """
        theta = np.asarray(theta_deg, dtype=float)
        theta_rad = np.radians(theta)
        cos_t = np.cos(theta_rad)
        safe_cos = np.where(theta < 80.0, np.maximum(cos_t, 0.02), 0.02)
        iam_val = np.where(theta < 80.0, 1.0 - self.b0 * (1.0 / safe_cos - 1.0), 0.0)
        return np.clip(iam_val, 0.0, 1.0)

    # ------------------------------------------------------------------
    # U_L(DeltaT) — temperature-dependent loss coefficient
    # ------------------------------------------------------------------
    def U_L(self, delta_T):
        """
        Second-order loss coefficient.
        U_L(DeltaT) = a1 + a2 * DeltaT   [W/m2K]
        For vacuum insulation: a2 is very small (~0.012 W/m2K2).
        """
        dT = np.asarray(delta_T, dtype=float)
        return self.a1 + self.a2 * np.maximum(dT, 0.0)

    # ------------------------------------------------------------------
    # Efficiency (ISO 9806 second-order form)
    # ------------------------------------------------------------------
    def efficiency(self, irradiance, T_inlet_c, T_ambient_c, theta_deg=0.0):
        """
        Instantaneous efficiency from ISO 9806 second-order equation.
        eta = eta_0 * IAM - a1 * dT_m / G - a2 * dT_m^2 / G
        where dT_m = T_m - T_amb = (T_in + T_out)/2 - T_amb.

        Note: This is the mean-temperature form; we iterate to find T_out.
        """
        G = np.asarray(irradiance, dtype=float)
        T_in = np.asarray(T_inlet_c, dtype=float)
        T_amb = np.asarray(T_ambient_c, dtype=float)
        iam_val = self.iam(theta_deg)

        # Iterative solve: start with T_out estimate = T_in + small gain
        T_out = T_in.copy() if hasattr(T_in, 'copy') else np.asarray(T_in + 5.0)
        T_out = np.asarray(T_out, dtype=float)

        for _ in range(10):
            T_m = 0.5 * (T_in + T_out)
            dT_m = T_m - T_amb
            eta = np.where(
                G > 1.0,
                iam_val * self.eta_0 - self.a1 * dT_m / G - self.a2 * dT_m * dT_m / G,
                0.0,
            )
            eta = np.clip(eta, 0.0, self.eta_0)
            Q_u = eta * self.area * G
            T_out_new = T_in + Q_u / (self.m_dot * self.cp)
            if np.max(np.abs(T_out_new - T_out)) < 0.01:
                T_out = T_out_new
                break
            T_out = T_out_new

        return np.clip(eta, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Useful heat and outlet temperature
    # ------------------------------------------------------------------
    def predict_all(self, irradiance, T_inlet_c, T_ambient_c, theta_deg=0.0):
        G = np.asarray(irradiance, dtype=float)
        T_in = np.asarray(T_inlet_c, dtype=float)
        T_amb = np.asarray(T_ambient_c, dtype=float)
        iam_val = self.iam(theta_deg)

        T_out = np.where(G > 1.0, T_in + 5.0, T_in)
        T_out = np.asarray(T_out, dtype=float)

        for _ in range(10):
            T_m = 0.5 * (T_in + T_out)
            dT_m = T_m - T_amb
            eta_raw = np.where(
                G > 1.0,
                iam_val * self.eta_0 - self.a1 * dT_m / G - self.a2 * dT_m * dT_m / G,
                0.0,
            )
            eta = np.clip(eta_raw, 0.0, self.eta_0)
            Q_u = eta * self.area * G
            T_out_new = T_in + Q_u / (self.m_dot * self.cp)
            if np.max(np.abs(T_out_new - T_out)) < 0.01:
                T_out = T_out_new
                break
            T_out = T_out_new

        T_m_final = 0.5 * (T_in + T_out)
        dT_m_final = T_m_final - T_amb
        U_L_eff = self.U_L(dT_m_final)

        return {
            "useful_heat_w": Q_u,
            "efficiency": eta,
            "T_outlet_c": T_out,
            "T_mean_c": T_m_final,
            "delta_T_m": dT_m_final,
            "U_L_eff_w_m2k": U_L_eff,
            "iam": np.asarray(iam_val * np.ones_like(G)),
        }
