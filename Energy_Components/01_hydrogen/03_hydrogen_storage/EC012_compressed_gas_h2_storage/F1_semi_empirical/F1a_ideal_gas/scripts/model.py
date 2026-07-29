"""
EC012 — Compressed Gas H2 Storage — F1a Ideal Gas Model

Real-gas equation:  PV = n Z R T
    where Z(P) = 1 + b1*P + b2*P^2  (compressibility factor)

Stored hydrogen mass:
    m_H2 = P * V / (Z * R_specific * T)    [kg]

Energy stored:
    E_stored = m_H2 * LHV_H2               [MJ]

Compression energy (isentropic with efficiency):
    W_comp = (P1*V1 / (k-1)) * ((P2/P1)^((k-1)/k) - 1) / eta_s
    Or per unit mass:
    w_comp = (k / (k-1)) * R_specific * T1 * ((P2/P1)^((k-1)/k) - 1) / eta_s   [J/kg]

References:
    Lemmon et al. (2008). NIST Chemistry WebBook.
    Zheng et al. (2012). Int. J. Hydrogen Energy, 37(2), 1048-1057.
"""

import numpy as np

R_UNIVERSAL = 8.314  # J/(mol·K)


class CompressedGasH2F1a:
    """Compressed gas hydrogen storage model with real-gas compressibility."""

    def __init__(self, params: dict):
        tank = params["tank"]["type_IV"]
        h2 = params["hydrogen"]
        comp = params["compressor"]
        z_params = params["compressibility"]

        self.V_tank = tank["volume"]["value"]           # m3
        self.P_max = tank["max_pressure"]["value"]      # bar
        self.P_min = tank["min_pressure"]["value"]      # bar
        self.tank_mass = tank["mass_empty"]["value"]    # kg

        self.M_H2 = h2["molar_mass"]["value"]          # kg/mol
        self.LHV = h2["LHV"]["value"]                  # MJ/kg
        self.gamma = h2["gamma"]["value"]               # cp/cv

        self.eta_s = comp["eta_isentropic"]["value"]
        self.T_inlet = comp["T_inlet"]["value"]         # K
        self.P_inlet = comp["P_inlet"]["value"]         # bar

        self.b1 = z_params["b1"]
        self.b2 = z_params["b2"]

        # Specific gas constant for H2
        self.R_specific = R_UNIVERSAL / self.M_H2       # J/(kg·K)

    def compressibility_factor(self, P_bar):
        """Compressibility factor Z(P). P in bar."""
        P = np.asarray(P_bar, dtype=float)
        return 1.0 + self.b1 * P + self.b2 * P**2

    def stored_mass(self, P_bar, T_K):
        """
        Mass of H2 stored [kg].

        Args:
            P_bar: Tank pressure [bar]
            T_K:   Tank temperature [K]
        """
        P = np.asarray(P_bar, dtype=float)
        T = np.asarray(T_K, dtype=float)
        P_Pa = P * 1e5
        Z = self.compressibility_factor(P)
        return P_Pa * self.V_tank / (Z * self.R_specific * T)

    def energy_stored(self, P_bar, T_K):
        """Energy stored [MJ] based on LHV of stored H2 mass."""
        m = self.stored_mass(P_bar, T_K)
        return m * self.LHV

    def compression_work(self, P1_bar, P2_bar, T1_K=None):
        """
        Specific compression work [kJ/kg_H2] for isentropic compression.

        Args:
            P1_bar: Inlet pressure [bar]
            P2_bar: Outlet pressure [bar]
            T1_K:   Inlet temperature [K] (default: self.T_inlet)
        """
        P1 = np.asarray(P1_bar, dtype=float)
        P2 = np.asarray(P2_bar, dtype=float)
        T1 = self.T_inlet if T1_K is None else np.asarray(T1_K, dtype=float)

        k = self.gamma
        ratio = (P2 / P1) ** ((k - 1.0) / k)
        w = (k / (k - 1.0)) * self.R_specific * T1 * (ratio - 1.0) / self.eta_s
        return w / 1000.0  # J/kg -> kJ/kg

    def fill_fraction(self, P_bar, T_K):
        """Fill fraction (0-1) = m_stored / m_stored_at_max_pressure."""
        m = self.stored_mass(P_bar, T_K)
        m_max = self.stored_mass(self.P_max, T_K)
        return np.clip(m / m_max, 0.0, 1.0)

    def gravimetric_density(self, P_bar, T_K):
        """Gravimetric storage density [wt%] = m_H2 / (m_H2 + m_tank) * 100."""
        m_h2 = self.stored_mass(P_bar, T_K)
        return m_h2 / (m_h2 + self.tank_mass) * 100.0

    def volumetric_density(self, P_bar, T_K):
        """Volumetric storage density [kg_H2/m3]."""
        return self.stored_mass(P_bar, T_K) / self.V_tank
