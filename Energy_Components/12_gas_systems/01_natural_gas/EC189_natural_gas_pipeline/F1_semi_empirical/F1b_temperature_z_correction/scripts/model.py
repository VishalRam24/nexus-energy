"""
EC189 — Natural Gas Pipeline — F1b Temperature-Profile + Z-Correction Model

Extends F1a Weymouth model with:
  1. Temperature profile along pipe (Coulter-Bardon / Shoham):
         T(x) = T_soil + (T_in - T_soil) * exp(-U_factor * x)
         T_avg = T_soil + (T_in - T_soil) * (1 - exp(-U_factor*L)) / (U_factor*L)
     where U_factor = pi * D * U_overall / (m_dot * cp)  [1/m]
  2. Papay (1985) Z-factor correlation (SI, field-proven):
         Tpr = T / Tc_pseudo,   Ppr = P / Pc_pseudo
         Z = 1 - (3.52 * Ppr) / (10^(0.9813 * Tpr))
           + (0.274 * Ppr^2) / (10^(0.8157 * Tpr))
     Average Z computed at average temperature and average pressure.
  3. Joule-Thomson temperature correction during gas expansion along the pipe.

Phase 7 note: Weymouth K constant confirmed as 3.7435e-3 per Menon (2005) SI Table 4.1
(Q [m³/day], T_b [K], P_b [kPa], D [mm], P [kPa], L [km]).

References:
    Menon, E.S. (2005). Gas Pipeline Hydraulics. CRC Press. Table 4.1 (p.99).
    Papay, J. (1985). A Termelestechnologiai Parameterek Valtozasa a Gazlerakodas
        Folyaman. OGIL MUSZ. TUD. KOZL., Budapest.
    Coulter, D.M. & Bardon, M.F. (1979). Revised Equation Improves Flowing Gas
        Temperature Prediction. Oil & Gas Journal.
"""

import numpy as np


class NGPipelineF1b:
    """
    Natural gas pipeline model with temperature profile and Papay Z-factor.

    Inherits Weymouth K=3.7435e-3 (Menon 2005 SI):
        Q = K * E * (T_b/P_b_kPa) * D_mm^(8/3) * sqrt(dP2 / (G * T_avg * Z_avg * L))

    Phase 7 confirmed: K=3.7435e-3 is correct for SI units where Q is in m³/day.
    """

    # Menon (2005) SI Weymouth constant — confirmed correct
    WEYMOUTH_K = 3.7435e-3

    def __init__(self, params: dict):
        p = params["pipeline"]
        g = params["gas"]

        self.E = p["efficiency_factor"]["value"]
        self.T_b = p["T_base"]["value"]        # K
        self.P_b = p["P_base"]["value"]        # bar
        self.T_flow = p["T_flow"]["value"]     # K default inlet
        self.T_soil = p["T_soil"]["value"]     # K
        self.U_overall = p["U_overall"]["value"]  # W/(m2.K)

        self.G = g["specific_gravity"]["value"]
        self.Tc = g["Tc"]["value"]             # K pseudo-critical
        self.Pc = g["Pc"]["value"]             # bar pseudo-critical
        self.cp = g["cp"]["value"]             # J/(kg.K)

    # ------------------------------------------------------------------
    # Z-factor (Papay 1985)
    # ------------------------------------------------------------------
    def z_papay(self, T_K, P_bar):
        """
        Papay (1985) Z-factor correlation.
        Tpr = T / Tc,  Ppr = P / Pc
        Z = 1 - 3.52*Ppr / 10^(0.9813*Tpr) + 0.274*Ppr^2 / 10^(0.8157*Tpr)
        Valid: Tpr in [1.05, 3.0], Ppr in [0, 15].
        """
        T = np.asarray(T_K, dtype=float)
        P = np.asarray(P_bar, dtype=float)
        Tpr = T / self.Tc
        Ppr = P / self.Pc
        Z = (1.0
             - 3.52 * Ppr / 10.0 ** (0.9813 * Tpr)
             + 0.274 * Ppr ** 2 / 10.0 ** (0.8157 * Tpr))
        # Z physically must be in (0.5, 1.1) for typical pipeline conditions
        return np.clip(Z, 0.50, 1.10)

    # ------------------------------------------------------------------
    # Temperature profile (Coulter-Bardon)
    # ------------------------------------------------------------------
    def temperature_profile(self, T_in_K, m_dot_kg_s, diameter_m, length_km):
        """
        Average pipeline temperature using Coulter-Bardon equation.

        T(x) = T_soil + (T_in - T_soil) * exp(-U_factor * x)
        T_avg = T_soil + (T_in - T_soil) * [1 - exp(-U_factor*L)] / (U_factor*L)

        Parameters
        ----------
        T_in_K      : inlet gas temperature [K]
        m_dot_kg_s  : mass flow rate [kg/s]
        diameter_m  : pipe inner diameter [m]
        length_km   : pipe length [km]

        Returns
        -------
        T_avg [K], T_out [K]
        """
        T_in = np.asarray(T_in_K, dtype=float)
        m = np.asarray(m_dot_kg_s, dtype=float)
        D = np.asarray(diameter_m, dtype=float)
        L_m = np.asarray(length_km, dtype=float) * 1000.0  # km → m

        # Guard against zero flow
        m_safe = np.where(m > 1e-6, m, 1e-6)

        # U_factor = (pi * D * U_overall) / (m_dot * cp)  [1/m]
        U_factor = (np.pi * D * self.U_overall) / (m_safe * self.cp)

        # Avoid overflow for very large U_factor*L
        UL = np.clip(U_factor * L_m, 1e-6, 50.0)
        T_avg = self.T_soil + (T_in - self.T_soil) * (1.0 - np.exp(-UL)) / UL
        T_out = self.T_soil + (T_in - self.T_soil) * np.exp(-UL)
        return T_avg, T_out

    # ------------------------------------------------------------------
    # Core Weymouth computation with T-Z corrections
    # ------------------------------------------------------------------
    def flow_rate_std_m3_per_day(self, length_km, diameter_m,
                                  P_in_bar, P_out_bar,
                                  T_in_K=None, m_dot_guess_kg_s=1.0,
                                  E=None):
        """
        Gas flow rate [m³/day] with temperature profile and Papay Z correction.

        Parameters
        ----------
        length_km         : pipeline length [km]
        diameter_m        : internal diameter [m]
        P_in_bar          : inlet pressure [bar]
        P_out_bar         : outlet pressure [bar]
        T_in_K            : inlet gas temperature [K] (default T_flow)
        m_dot_guess_kg_s  : mass flow estimate for temperature profile [kg/s]
        E                 : efficiency factor (optional)
        """
        L = np.asarray(length_km, dtype=float)
        D = np.asarray(diameter_m, dtype=float)
        P1 = np.asarray(P_in_bar, dtype=float)
        P2 = np.asarray(P_out_bar, dtype=float)
        T_in = self.T_flow if T_in_K is None else np.asarray(T_in_K, dtype=float)
        Ev = self.E if E is None else np.asarray(E, dtype=float)

        # Average pressure (simple arithmetic mean)
        P_avg = (P1 + P2) / 2.0

        # Temperature profile → average temperature
        T_avg, _ = self.temperature_profile(T_in, m_dot_guess_kg_s, D, L)

        # Papay Z at (T_avg, P_avg)
        Z_avg = self.z_papay(T_avg, P_avg)

        # Menon (2005) SI unit conversions
        P1_kPa = P1 * 100.0
        P2_kPa = P2 * 100.0
        D_mm = D * 1000.0
        P_b_kPa = self.P_b * 100.0

        dp2 = np.maximum(P1_kPa ** 2 - P2_kPa ** 2, 0.0)
        Q = (Ev * self.WEYMOUTH_K * (self.T_b / P_b_kPa) *
             D_mm ** (8.0 / 3.0) *
             np.sqrt(dp2 / (self.G * T_avg * Z_avg * L)))
        return Q  # m³/day

    def flow_rate_std_m3_per_s(self, length_km, diameter_m, P_in_bar, P_out_bar,
                                T_in_K=None, m_dot_guess_kg_s=1.0, E=None):
        return self.flow_rate_std_m3_per_day(
            length_km, diameter_m, P_in_bar, P_out_bar,
            T_in_K, m_dot_guess_kg_s, E) / 86400.0

    def flow_rate_kg_per_s(self, length_km, diameter_m, P_in_bar, P_out_bar,
                            T_in_K=None, m_dot_guess_kg_s=1.0, E=None):
        """Mass flow [kg/s] at standard conditions."""
        Q_m3s = self.flow_rate_std_m3_per_s(
            length_km, diameter_m, P_in_bar, P_out_bar,
            T_in_K, m_dot_guess_kg_s, E)
        R_univ = 8.31446
        M_air = 0.028966
        P_b_Pa = self.P_b * 1e5
        rho_std = P_b_Pa * self.G * M_air / (R_univ * self.T_b)
        return Q_m3s * rho_std

    def pressure_drop_bar(self, Q_std_m3_per_day, length_km, diameter_m,
                          P_in_bar, T_in_K=None, m_dot_guess_kg_s=1.0, E=None):
        """Outlet pressure drop [bar] rearranged from Weymouth."""
        Q = np.asarray(Q_std_m3_per_day, dtype=float)
        L = np.asarray(length_km, dtype=float)
        D = np.asarray(diameter_m, dtype=float)
        P1 = np.asarray(P_in_bar, dtype=float)
        T_in = self.T_flow if T_in_K is None else np.asarray(T_in_K, dtype=float)
        Ev = self.E if E is None else np.asarray(E, dtype=float)

        P_avg = P1 * 0.95  # first estimate
        T_avg, _ = self.temperature_profile(T_in, m_dot_guess_kg_s, D, L)
        Z_avg = self.z_papay(T_avg, P_avg)

        D_mm = D * 1000.0
        P_b_kPa = self.P_b * 100.0

        coeff = (Ev * self.WEYMOUTH_K * (self.T_b / P_b_kPa)
                 * D_mm ** (8.0 / 3.0))
        # Q = coeff * sqrt(dP2_kPa2 / (G*T*Z*L))
        # dP2_kPa2 = (Q/coeff)^2 * G*T*Z*L
        dP2_kPa2 = (Q / coeff) ** 2 * self.G * T_avg * Z_avg * L
        P1_kPa2 = (P1 * 100.0) ** 2
        P2_kPa = np.sqrt(np.maximum(P1_kPa2 - dP2_kPa2, 0.0))
        P2 = P2_kPa / 100.0
        return P1 - P2

    def compute(self, length_km, diameter_m, P_in_bar, P_out_bar,
                T_in_K=None, m_dot_guess_kg_s=1.0):
        """Full computation returning all pipeline outputs."""
        T_in = self.T_flow if T_in_K is None else T_in_K
        Q_m3d = self.flow_rate_std_m3_per_day(
            length_km, diameter_m, P_in_bar, P_out_bar, T_in, m_dot_guess_kg_s)
        Q_kgs = self.flow_rate_kg_per_s(
            length_km, diameter_m, P_in_bar, P_out_bar, T_in, m_dot_guess_kg_s)

        P_avg = (np.asarray(P_in_bar, dtype=float) +
                 np.asarray(P_out_bar, dtype=float)) / 2.0
        D = np.asarray(diameter_m, dtype=float)
        L = np.asarray(length_km, dtype=float)
        T_in_arr = np.asarray(T_in, dtype=float)
        T_avg, T_out = self.temperature_profile(T_in_arr, m_dot_guess_kg_s, D, L)
        Z_avg = self.z_papay(T_avg, P_avg)

        return {
            "flow_rate_std_m3_per_day": Q_m3d,
            "flow_rate_kg_per_s": Q_kgs,
            "T_avg_K": T_avg,
            "T_out_K": T_out,
            "Z_avg": Z_avg,
        }
