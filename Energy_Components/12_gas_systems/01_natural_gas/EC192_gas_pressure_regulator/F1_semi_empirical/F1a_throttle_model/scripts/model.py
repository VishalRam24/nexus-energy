"""
EC192 — Gas Pressure Regulator — F1a Throttle Model

Physics:
1. Pressure reduction is isenthalpic (Joule-Thomson process): h_up = h_down
2. Temperature change from JT effect:
       ΔT = μ_JT × ΔP     where ΔP = P_up - P_down  (in Pa)
       T_down = T_up - μ_JT × (P_up - P_down)
   For natural gas at typical conditions: μ_JT ≈ 4.5×10⁻⁶ K/Pa ≈ 0.45 K/bar
   Cooling on expansion (μ_JT > 0) for T < T_inversion (~700 K for NG)

3. Flow through control valve (ISA S75 gas sizing, subcritical flow):
       Q_std [m³/h] = N₇ × Cv × P_up × sqrt(ΔP / (G × T_up × Z)) / (P_up)
   Simplified ISA form (non-choked):
       Q = N₇ × Cv × Y × sqrt(ΔP × P_up / (G × T × Z))
   where Y = 1 - ΔP/(3 × Fk × Xt × P_up)  (expansion factor)
   and N₇ = 4.17×10⁻² (SI, Q in m³/h, P in bar, T in K)

   Choked when: ΔP ≥ Fk × Xt × P_up

References:
    ANSI/ISA-75.01.01-2012. Flow Equations for Sizing Control Valves.
    Burnett, R.R. (1999). Joule-Thomson Coefficients for Gas Mixtures.
    Engineering Toolbox: Joule-Thomson effect for natural gas.
"""

import numpy as np


class GasPressureRegulatorF1a:
    """Isenthalpic gas pressure regulator (JT throttle + Cv flow equation)."""

    # ISA N7 constant: Q [m³/h at std], P [bar], T [K]
    N7 = 4.17e-2

    def __init__(self, params: dict):
        v = params["valve"]
        g = params["gas"]

        self.Cv_default = v["Cv_default"]["value"]
        self.Xt = v["Xt_default"]["value"]
        self.Fk = v["Fk_default"]["value"]

        self.G = g["specific_gravity"]["value"]
        self.gamma = g["gamma"]["value"]
        self.R_s = g["R_specific"]["value"]
        self.cp = g["cp"]["value"]
        self.mu_JT = g["JT_coefficient"]["value"]    # K/Pa
        self.T_inv = g["T_inversion"]["value"]       # K

    def temperature_out(self, T_up_K, P_up_bar, P_down_bar):
        """
        Downstream temperature after isenthalpic expansion [K].
        ΔT = μ_JT × (P_up - P_down) [in Pa]
        """
        T_up = np.asarray(T_up_K, dtype=float)
        P_up = np.asarray(P_up_bar, dtype=float)
        P_down = np.asarray(P_down_bar, dtype=float)

        dP_Pa = (P_up - P_down) * 1e5  # bar → Pa
        mu = np.where(T_up < self.T_inv, self.mu_JT, -self.mu_JT)
        return T_up - mu * dP_Pa

    def is_choked(self, P_up_bar, P_down_bar):
        """True where flow is choked (critical pressure drop reached)."""
        P_up = np.asarray(P_up_bar, dtype=float)
        P_down = np.asarray(P_down_bar, dtype=float)
        dP = P_up - P_down
        return dP >= self.Fk * self.Xt * P_up

    def expansion_factor_Y(self, P_up_bar, P_down_bar):
        """
        ISA expansion factor Y (1 at ΔP=0, 0.667 at choke).
        Clamped to 0.667 when choked.
        """
        P_up = np.asarray(P_up_bar, dtype=float)
        P_down = np.asarray(P_down_bar, dtype=float)
        dP = P_up - P_down
        Y = 1.0 - dP / (3.0 * self.Fk * self.Xt * P_up)
        return np.maximum(Y, 2.0 / 3.0)

    def flow_std_m3_per_h(self, P_up_bar, P_down_bar, T_up_K, Z=0.9, Cv=None):
        """
        Gas flow at standard conditions [m³/h] via ISA Cv equation.

        Subcritical:  Q = N7 × Cv × Y × sqrt(ΔP × P_up / (G × T × Z))
        Choked:       Q = N7 × Cv × (2/3) × sqrt(ΔP_choke × P_up / (G × T × Z))
                         where ΔP_choke = Fk × Xt × P_up

        At choke, Y=2/3 and ΔP in sqrt is replaced by the critical ΔP_choke
        (not the actual ΔP), so flow is independent of downstream pressure.
        """
        P_up = np.asarray(P_up_bar, dtype=float)
        P_down = np.asarray(P_down_bar, dtype=float)
        T_up = np.asarray(T_up_K, dtype=float)
        Z = np.asarray(Z, dtype=float)
        Cv_val = self.Cv_default if Cv is None else np.asarray(Cv, dtype=float)

        dP_actual = np.maximum(P_up - P_down, 0.0)
        dP_choke = self.Fk * self.Xt * P_up
        choked = self.is_choked(P_up, P_down)

        # Use actual ΔP in subcritical; use choke ΔP when choked
        dP_eff = np.where(choked, dP_choke, dP_actual)
        Y = self.expansion_factor_Y(P_up, P_down)  # clamped to 2/3 at choke

        Q = self.N7 * Cv_val * Y * np.sqrt(dP_eff * P_up / (self.G * T_up * Z))
        return Q

    def flow_kg_per_s(self, P_up_bar, P_down_bar, T_up_K, Z=0.9, Cv=None):
        """Mass flow [kg/s]. rho_std ≈ 0.717 kg/m³ for NG at standard conditions."""
        Q_m3h = self.flow_std_m3_per_h(P_up_bar, P_down_bar, T_up_K, Z, Cv)
        rho_std = 0.717  # kg/m³ at 288.15 K, 1.01325 bar
        return Q_m3h * rho_std / 3600.0
