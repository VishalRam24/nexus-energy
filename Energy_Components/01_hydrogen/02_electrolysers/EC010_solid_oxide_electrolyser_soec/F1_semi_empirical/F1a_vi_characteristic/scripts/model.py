"""
EC010 — Solid Oxide Electrolyser (SOEC) — F1a V-I Characteristic

ASR-based semi-empirical model:

    V_cell = E_rev(T) + j * ASR(T)

where:
    E_rev(T) = 1.253 - 0.00024*(T - 298)    [V]  reversible potential for steam electrolysis
    ASR(T)   = ASR_ref * exp(E_act/R * (1/T - 1/T_ref))   [Ohm.cm2] Arrhenius-type

Endothermic vs. exothermic boundary: V_cell compared to thermo-neutral voltage V_tn ≈ 1.285 V (at 800 C)
    V_cell < V_tn  => endothermic operation (requires external heat)
    V_cell > V_tn  => exothermic operation (self-sustaining thermally)

Hydrogen production rate (Faraday's law, 100% Faraday efficiency assumed for SOEC):
    H2_rate = N_cells * j * A / (2*F)   [mol/s]

Reference:
    Ni et al. (2007), Chem. Eng. Tech., 29(6), 636-642.
"""

import numpy as np


class SOECF1a:
    """Solid Oxide Electrolyser — ASR-based V-I characteristic model."""

    def __init__(self, params: dict):
        u = params["unit"]
        c = params["constants"]

        self.N_cells = u["N_cells"]["value"]
        self.A_cm2 = u["electrode_area"]["value"]          # cm2
        self.A_m2 = self.A_cm2 * 1e-4                     # m2
        self.T_ref = u["T_ref"]["value"]                    # K
        self.ASR_ref = u["ASR_ref"]["value"]                # Ohm.cm2
        self.E_act = u["E_act"]["value"]                    # J/mol
        self.V_tn = u["E_thermo_neutral"]["value"]          # V

        self.F = c["F"]["value"]                            # C/mol
        self.R = c["R"]["value"]                            # J/(mol.K)
        self.E_rev_ref = c["E_rev_ref"]["value"]            # V at 298 K
        self.E_rev_T = c["E_rev_T_coeff"]["value"]          # V/K

    # ------------------------------------------------------------------ #
    # Core electrochemistry
    # ------------------------------------------------------------------ #

    def e_rev(self, T_K):
        """Reversible cell voltage for steam electrolysis as f(T) [V]."""
        T_K = np.asarray(T_K, dtype=float)
        return self.E_rev_ref - self.E_rev_T * (T_K - 298.15)

    def asr(self, T_K):
        """Area-specific resistance via Arrhenius correlation [Ohm.cm2]."""
        T_K = np.asarray(T_K, dtype=float)
        return self.ASR_ref * np.exp(self.E_act / self.R * (1.0 / T_K - 1.0 / self.T_ref))

    def cell_voltage(self, j_A_cm2, T_K):
        """
        Cell voltage [V].

        Args:
            j_A_cm2: Current density [A/cm2]
            T_K:     Temperature [K]

        Returns:
            V_cell [V]
        """
        j = np.asarray(j_A_cm2, dtype=float)
        T_K = np.asarray(T_K, dtype=float)
        return self.e_rev(T_K) + j * self.asr(T_K)

    def stack_voltage(self, j_A_cm2, T_K):
        """Total stack voltage [V]."""
        return self.N_cells * self.cell_voltage(j_A_cm2, T_K)

    def thermal_mode(self, j_A_cm2, T_K):
        """
        Thermal operation mode.
        Returns: 1 = endothermic (V < V_tn), 0 = thermo-neutral, -1 = exothermic.
        """
        V = self.cell_voltage(j_A_cm2, T_K)
        return np.sign(self.V_tn - V).astype(int)

    def hydrogen_rate(self, j_A_cm2):
        """
        Hydrogen production rate [mol/s].
        Assumes 100% Faraday efficiency (typical for SOEC at operating conditions).
        """
        j = np.asarray(j_A_cm2, dtype=float)
        I = j * self.A_cm2   # total current [A]
        return self.N_cells * I / (2.0 * self.F)

    def power_kw(self, j_A_cm2, T_K):
        """Electrical power consumed by the stack [kW]."""
        j = np.asarray(j_A_cm2, dtype=float)
        I = j * self.A_cm2
        V_stack = self.stack_voltage(j, T_K)
        return V_stack * I / 1000.0

    def efficiency(self, j_A_cm2, T_K):
        """
        Stack electrical efficiency = (H2 LHV chemical energy rate) / (electrical power).
        LHV of H2 = 241.8 kJ/mol.
        """
        j = np.asarray(j_A_cm2, dtype=float)
        H2_LHV = 241800.0  # J/mol
        n_H2 = self.hydrogen_rate(j)
        p_el = self.power_kw(j, T_K) * 1000.0
        p_chem = n_H2 * H2_LHV
        safe_p_el = np.where(p_el > 0, p_el, 1.0)
        return np.where(p_el > 0, np.clip(p_chem / safe_p_el, 0.0, 1.5), 0.0)
