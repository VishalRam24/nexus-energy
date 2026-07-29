"""
EC009 — Alkaline Electrolyser (AEL) — F1a V-I Characteristic

Ulleberg (2003) semi-empirical model for alkaline water electrolysis:

    V_cell = E_rev(T) + r(T)/A * j + s * log10((t1 + t2/T + t3/T^2) * j/A + 1)

where:
    E_rev(T) = 1.229 - 0.0009*(T - 298)   [V] reversible cell voltage
    r(T)     = r1 + r2*T                   [Ohm.m2] ohmic resistance
    j        = current density             [A/m2]
    A        = electrode area              [m2]

Hydrogen production rate (Faraday's law):
    H2_rate = eta_F * N_cells * j * A / (2*F)   [mol/s]

Faraday efficiency:
    eta_F = (f1 * (j*A/1000)^2) / (f2 + (j*A/1000)^2)

Reference:
    Ulleberg (2003), Int. J. Hydrogen Energy, 28(1), 21-33.
"""

import numpy as np


class AELF1a:
    """Alkaline Electrolyser — Ulleberg (2003) V-I characteristic model."""

    def __init__(self, params: dict):
        u = params["unit"]
        c = params["constants"]

        self.N_cells = u["N_cells"]["value"]
        self.A = u["electrode_area"]["value"]          # m2
        self.r1 = u["r1"]["value"]                     # Ohm.m2
        self.r2 = u["r2"]["value"]                     # Ohm.m2/K
        self.s = u["s"]["value"]                       # V
        self.t1 = u["t1"]["value"]                     # m2/A
        self.t2 = u["t2"]["value"]                     # m2.K/A
        self.t3 = u["t3"]["value"]                     # m2.K2/A
        self.f1 = u["f1"]["value"]                     # (mA/cm2)^2  [dimensionless numerator]
        self.f2 = u["f2"]["value"]                     # dimensionless
        self.F = c["F"]["value"]                       # C/mol
        self.E_rev_ref = c["E_rev_ref"]["value"]       # V at 25 C
        self.E_rev_T = c["E_rev_T_coeff"]["value"]     # V/K

    # ------------------------------------------------------------------ #
    # Core electrochemistry
    # ------------------------------------------------------------------ #

    def e_rev(self, T_K):
        """Reversible (Nernst) cell voltage as a function of temperature."""
        T_K = np.asarray(T_K, dtype=float)
        return self.E_rev_ref - self.E_rev_T * (T_K - 298.15)

    def ohmic_resistance(self, T_K):
        """Temperature-dependent ohmic resistance [Ohm.m2]."""
        T_K = np.asarray(T_K, dtype=float)
        return self.r1 + self.r2 * T_K

    def cell_voltage(self, j, T_K):
        """
        Cell terminal voltage using Ulleberg (2003) model.

        Args:
            j:   Current density [A/m2]
            T_K: Temperature [K]

        Returns:
            V_cell [V]
        """
        j = np.asarray(j, dtype=float)
        T_K = np.asarray(T_K, dtype=float)

        E_rev = self.e_rev(T_K)
        r = self.ohmic_resistance(T_K)

        # Overvoltage term — protect log against j=0
        arg = (self.t1 + self.t2 / T_K + self.t3 / T_K**2) * (j / self.A) + 1.0
        arg = np.where(j > 0, arg, 1.0)

        V_ohm = (r / self.A) * j
        V_act = np.where(j > 0, self.s * np.log10(arg), 0.0)

        return E_rev + V_ohm + V_act

    def stack_voltage(self, j, T_K):
        """Total stack voltage [V]."""
        return self.N_cells * self.cell_voltage(j, T_K)

    def faraday_efficiency(self, j):
        """
        Current efficiency (Faraday efficiency) [-].

        Args:
            j: Current density [A/m2]

        Returns:
            eta_F in [0, 1]
        """
        j = np.asarray(j, dtype=float)
        # Convert to mA/cm2 for consistency with Ulleberg f1 units
        j_ma_cm2 = j * self.A / 1000.0   # total current I in A, then /1000 for mA
        # Actually Ulleberg uses I [A] directly — f1 in (A)^2:
        # eta_F = f1*I^2 / (f2 + I^2), I = j*A
        I = j * self.A
        eta_F = (self.f1 * I**2) / (self.f2 + I**2)
        return np.where(j > 0, np.clip(eta_F, 0.0, 1.0), 0.0)

    def hydrogen_rate(self, j, T_K):
        """
        Hydrogen production rate [mol/s].

        Uses Faraday's law corrected by Faraday efficiency.
        """
        j = np.asarray(j, dtype=float)
        eta_F = self.faraday_efficiency(j)
        I = j * self.A
        return eta_F * self.N_cells * I / (2.0 * self.F)

    def power_kw(self, j, T_K):
        """Electrical power consumed by the stack [kW]."""
        V_stack = self.stack_voltage(j, T_K)
        I = np.asarray(j, dtype=float) * self.A
        return V_stack * I / 1000.0

    def efficiency(self, j, T_K):
        """
        Stack efficiency = (H2 chemical energy) / (electrical input).
        Uses LHV of H2 = 241.8 kJ/mol.
        """
        j = np.asarray(j, dtype=float)
        H2_LHV = 241800.0  # J/mol
        n_H2 = self.hydrogen_rate(j, T_K)          # mol/s
        p_el = self.power_kw(j, T_K) * 1000.0      # W
        p_chem = n_H2 * H2_LHV                     # W
        safe_p_el = np.where(p_el > 0, p_el, 1.0)
        return np.where(p_el > 0, np.clip(p_chem / safe_p_el, 0.0, 1.0), 0.0)
