"""
EC189 — Natural Gas Pipeline — F1a Weymouth Model

Weymouth equation for isothermal, steady-state gas flow in a pipeline.

Standard Menon (2005) SI form (Equation 4.19, Table 4.1):

    Q = 3.7435e-3 × E × (T_b/P_b) × D^(8/3) × sqrt((P1²-P2²) / (G × T_f × Z × L))

where:
    Q   = gas flow rate at standard conditions [m³/day]
    E   = pipeline efficiency factor [-]
    T_b = base temperature [K]            (288.15 K)
    P_b = base pressure [kPa]            (101.325 kPa)
    D   = internal pipe diameter [mm]
    P1  = inlet pressure [kPa]
    P2  = outlet pressure [kPa]
    G   = gas specific gravity (air=1) [-]
    T_f = average gas temperature [K]
    Z   = average compressibility factor [-]
    L   = pipeline length [km]

Interface inputs use bar and metres; the model converts internally to kPa and mm.

Weymouth friction factor (implicit in constant 3.7435e-3):
    f_w = 0.032 / D_mm^(1/3)  (D in mm)

Flow scaling: Q ∝ √(P1² - P2²)

References:
    Weymouth, T.R. (1912). Problems in Natural Gas Engineering. Trans ASME, 34, 185-231.
    Menon, E.S. (2005). Gas Pipeline Hydraulics. CRC Press. Chapter 4, Eq. 4.19.
"""

import numpy as np


class NGPipelineF1a:
    """Weymouth equation natural gas pipeline model.

    User-facing units: P in bar, L in km, D in m.
    Internal computation: P in kPa, D in mm (per Menon 2005 SI formulation).
    """

    # Menon (2005) Table 4.1 SI constant: Q [m³/day], T_b [K], P_b [kPa], D [mm], P [kPa], L [km]
    WEYMOUTH_K = 3.7435e-3

    def __init__(self, params: dict):
        p = params["pipeline"]
        g = params["gas"]

        self.E = p["efficiency_factor"]["value"]
        self.T_b = p["T_base"]["value"]       # K
        self.P_b = p["P_base"]["value"]       # bar
        self.T_flow = p["T_flow"]["value"]    # K
        self.G = g["specific_gravity"]["value"]
        self.Z = g["Z_avg"]["value"]

    def flow_rate_std_m3_per_day(self, length_km, diameter_m,
                                  P_in_bar, P_out_bar,
                                  T_K=None, Z=None, E=None):
        """
        Compute gas flow rate via Weymouth equation.

        Args:
            length_km   : pipeline length [km]
            diameter_m  : internal diameter [m]
            P_in_bar    : inlet pressure [bar]
            P_out_bar   : outlet pressure [bar]
            T_K         : average gas temperature [K] (optional, defaults to T_flow)
            Z           : compressibility factor [-] (optional, defaults to Z_avg)
            E           : efficiency factor [-] (optional, defaults to params value)

        Returns:
            Q [m³/day at standard conditions]
        """
        L = np.asarray(length_km, dtype=float)
        D = np.asarray(diameter_m, dtype=float)
        P1 = np.asarray(P_in_bar, dtype=float)
        P2 = np.asarray(P_out_bar, dtype=float)
        T = self.T_flow if T_K is None else np.asarray(T_K, dtype=float)
        Zv = self.Z if Z is None else np.asarray(Z, dtype=float)
        Ev = self.E if E is None else np.asarray(E, dtype=float)

        # Convert to Menon (2005) SI units: P bar→kPa, D m→mm
        P1_kPa = P1 * 100.0
        P2_kPa = P2 * 100.0
        D_mm = D * 1000.0
        P_b_kPa = self.P_b * 100.0  # P_b in kPa (101.325 kPa)

        dp2 = np.maximum(P1_kPa ** 2 - P2_kPa ** 2, 0.0)
        Q = (Ev * self.WEYMOUTH_K * (self.T_b / P_b_kPa) *
             D_mm ** (8.0 / 3.0) *
             np.sqrt(dp2 / (self.G * T * Zv * L)))
        return Q  # m³/day standard

    def flow_rate_std_m3_per_s(self, length_km, diameter_m, P_in_bar, P_out_bar,
                                T_K=None, Z=None, E=None):
        return self.flow_rate_std_m3_per_day(
            length_km, diameter_m, P_in_bar, P_out_bar, T_K, Z, E) / 86400.0

    def flow_rate_kg_per_s(self, length_km, diameter_m, P_in_bar, P_out_bar,
                            T_K=None, Z=None, E=None):
        """Mass flow at standard conditions (rho_std = P_b * G * M_air / (Z_b * R * T_b))."""
        Q_m3s = self.flow_rate_std_m3_per_s(
            length_km, diameter_m, P_in_bar, P_out_bar, T_K, Z, E)
        # Density at standard conditions: ideal gas, Z=1 at base
        R_universal = 8.31446  # J/(mol K)
        M_air = 0.028966       # kg/mol
        G = self.G
        P_b_Pa = self.P_b * 1e5
        rho_std = (P_b_Pa * G * M_air) / (R_universal * self.T_b)  # kg/m³
        return Q_m3s * rho_std

    def pressure_drop_bar(self, Q_std_m3_per_day, length_km, diameter_m,
                          P_in_bar, T_K=None, Z=None, E=None):
        """Solve for P_out given Q (Weymouth rearranged)."""
        Q = np.asarray(Q_std_m3_per_day, dtype=float)
        L = np.asarray(length_km, dtype=float)
        D = np.asarray(diameter_m, dtype=float)
        P1 = np.asarray(P_in_bar, dtype=float)
        T = self.T_flow if T_K is None else np.asarray(T_K, dtype=float)
        Zv = self.Z if Z is None else np.asarray(Z, dtype=float)
        Ev = self.E if E is None else np.asarray(E, dtype=float)

        coeff = Ev * self.WEYMOUTH_K * (self.T_b / self.P_b) * D ** (8.0 / 3.0)
        # Q = coeff * sqrt(dP2/(G*T*Z*L))  =>  dP2 = (Q/coeff)^2 * G*T*Z*L
        dP2 = (Q / coeff) ** 2 * self.G * T * Zv * L
        P2_sq = np.maximum(P1 ** 2 - dP2, 0.0)
        P2 = np.sqrt(P2_sq)
        return P1 - P2  # pressure drop [bar]

    def weymouth_friction_factor(self, diameter_m):
        """Weymouth implicit friction factor: f = 0.032 / D^(1/3)."""
        D = np.asarray(diameter_m, dtype=float)
        return 0.032 / D ** (1.0 / 3.0)
