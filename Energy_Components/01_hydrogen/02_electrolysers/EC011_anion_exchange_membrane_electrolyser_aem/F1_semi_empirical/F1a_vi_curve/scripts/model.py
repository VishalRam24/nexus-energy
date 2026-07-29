"""
EC011 — Anion Exchange Membrane Electrolyser (AEM) — F1a V-I Polarization Curve

Semi-empirical Tafel + ohmic polarization model:

    V_cell = E_rev(T) + V_act,a + V_act,c + V_ohm

where:
    E_rev(T)  = 1.229 - 0.0009*(T - 298.15)               [V]
    V_act,a   = (R*T)/(alpha_a*F) * ln(j / j0_a)          [V]   (Tafel anode)
    V_act,c   = (R*T)/(alpha_c*F) * ln(j / j0_c)          [V]   (Tafel cathode)
    V_ohm     = ASR(T) * j                                [V]
    ASR(T)    = r_ref * (1 + r_T*(T - T_ref))             [Ohm.cm2]

Hydrogen production (Faraday's law):
    n_H2 = eta_F * N_cells * I / (2*F)                    [mol/s]

References:
    Vincent & Bessarabov (2018) Renew. Sustain. Energy Rev., 81, 1690.
    Henkensmeier et al. (2021) J. Electrochem. Energy Conv. Storage, 18, 024001.
"""

import numpy as np


class AEMF1a:
    """Anion Exchange Membrane Electrolyser — Tafel + ohmic polarization model."""

    def __init__(self, params: dict):
        u = params["unit"]
        c = params["constants"]

        self.N_cells = u["N_cells"]["value"]
        self.A_m2 = u["electrode_area"]["value"]            # m2
        self.A_cm2 = self.A_m2 * 1.0e4                       # cm2

        self.j0_a_ref = u["j0_anode"]["value"]               # A/cm2 at T_ref
        self.j0_c_ref = u["j0_cathode"]["value"]             # A/cm2 at T_ref
        self.Ea_a = u["Ea_anode"]["value"]                   # J/mol
        self.Ea_c = u["Ea_cathode"]["value"]                 # J/mol
        self.alpha_a = u["alpha_anode"]["value"]
        self.alpha_c = u["alpha_cathode"]["value"]

        self.r_ref = u["r_membrane_ref"]["value"]            # Ohm.cm2
        self.r_T = u["r_temp_coeff"]["value"]                # 1/K
        self.T_ref = u["T_ref"]["value"]                     # K
        self.eta_F = u["eta_F"]["value"]

        self.F = c["F"]["value"]
        self.R = c["R"]["value"]
        self.E_rev_ref = c["E_rev_ref"]["value"]
        self.E_rev_T = c["E_rev_T_coeff"]["value"]

    # ------------------------------------------------------------------ #
    def e_rev(self, T_K):
        T_K = np.asarray(T_K, dtype=float)
        return self.E_rev_ref - self.E_rev_T * (T_K - 298.15)

    def asr(self, T_K):
        """Area specific resistance [Ohm.cm2]."""
        T_K = np.asarray(T_K, dtype=float)
        return self.r_ref * (1.0 + self.r_T * (T_K - self.T_ref))

    def cell_voltage(self, j_A_m2, T_K):
        """
        Cell voltage [V].

        Args:
            j_A_m2: current density [A/m2]
            T_K:    temperature [K]
        """
        j = np.asarray(j_A_m2, dtype=float)
        T_K = np.asarray(T_K, dtype=float)

        # Convert to A/cm2 for Tafel terms
        j_cm2 = j / 1.0e4
        # Avoid log(0)
        j_safe = np.where(j_cm2 > 1e-12, j_cm2, 1e-12)

        E_rev = self.e_rev(T_K)
        RT_F = self.R * T_K / self.F

        # Arrhenius temperature dependence of exchange current densities
        j0_a = self.j0_a_ref * np.exp(-self.Ea_a / self.R * (1.0 / T_K - 1.0 / self.T_ref))
        j0_c = self.j0_c_ref * np.exp(-self.Ea_c / self.R * (1.0 / T_K - 1.0 / self.T_ref))

        V_act_a = np.where(j_cm2 > 1e-12,
                           (RT_F / self.alpha_a) * np.log(j_safe / j0_a), 0.0)
        V_act_c = np.where(j_cm2 > 1e-12,
                           (RT_F / self.alpha_c) * np.log(j_safe / j0_c), 0.0)
        # Activation overpotentials must be >= 0 (positive when j > j0)
        V_act_a = np.maximum(V_act_a, 0.0)
        V_act_c = np.maximum(V_act_c, 0.0)

        ASR = self.asr(T_K)            # Ohm.cm2
        V_ohm = ASR * j_cm2            # Volts

        return E_rev + V_act_a + V_act_c + V_ohm

    def stack_voltage(self, j, T_K):
        return self.N_cells * self.cell_voltage(j, T_K)

    def hydrogen_rate(self, j, T_K=None):
        """Hydrogen production [mol/s]."""
        j = np.asarray(j, dtype=float)
        I = j * self.A_m2  # Amps
        return self.eta_F * self.N_cells * I / (2.0 * self.F)

    def power_kw(self, j, T_K):
        V_stack = self.stack_voltage(j, T_K)
        I = np.asarray(j, dtype=float) * self.A_m2
        return V_stack * I / 1000.0

    def efficiency(self, j, T_K):
        """LHV efficiency = (H2 LHV power) / (electrical power)."""
        H2_LHV = 241800.0  # J/mol
        n_H2 = self.hydrogen_rate(j, T_K)
        p_el = self.power_kw(j, T_K) * 1000.0
        safe = np.where(p_el > 0, p_el, 1.0)
        return np.where(p_el > 0, np.clip(n_H2 * H2_LHV / safe, 0.0, 1.0), 0.0)
