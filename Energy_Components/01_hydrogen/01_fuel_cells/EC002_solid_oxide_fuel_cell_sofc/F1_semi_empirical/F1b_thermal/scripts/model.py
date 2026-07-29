"""
EC002 -- Solid Oxide Fuel Cell (SOFC) -- F1b Thermal Model
Temperature-dependent polarization with Arrhenius kinetics and YSZ conductivity.

Physics:
  - Nernst: E(T) = -dG(T)/(nF) approximated as E0(T) + RT/(2F)*ln(pH2*sqrt(pO2)/pH2O)
  - YSZ ionic conductivity: sigma_ion(T) = (A_sigma/T)*exp(-E_act_ion/(R*T))
  - Ohmic ASR: R_ohm = thickness / sigma_ion(T)
  - Anode activation: i0_a(T) = i0_a_ref * exp(-E_act_a/R*(1/T - 1/T_ref))
  - Cathode activation: i0_c(T) = i0_c_ref * exp(-E_act_c/R*(1/T - 1/T_ref))
  - V_act = RT/(alpha*n*F) * [arcsinh(j/(2*i0_a)) + arcsinh(j/(2*i0_c))]
  - Concentration: V_conc = -RT/(nF)*ln(1 - j/j_L)
  - Heat: Q = j*(E_tn - V_cell), E_tn ~ 1.285 V

References:
    Chan et al. (2001), J. Power Sources, 93, 130-140
    Bessette et al. (1995), J. Electrochem. Soc., 142(11), 3792
    Virkar (2005), J. Power Sources, 147(1-2), 125-136
"""

import numpy as np


class SOFCThermalModel:
    """
    Solid oxide fuel cell with explicit temperature dependence.
    Operating range: 973-1273 K (700-1000 C).
    """

    R = 8.314
    F = 96485.0
    n = 2
    E_tn = 1.285  # thermoneutral voltage at ~800C [V]
    LHV_H2 = 241800.0  # J/mol

    def __init__(self, params: dict):
        self.T_ref = float(params["T_ref"])
        self.N_cells = int(params["N_cells"])
        self.A_cell = float(params["A_cell"])
        self.pH2 = float(params["pH2"])
        self.pO2 = float(params["pO2"])
        self.pH2O = float(params["pH2O"])
        self.j_L = float(params["j_L"])
        self.A_sigma = float(params["A_sigma"])
        self.E_act_ion = float(params["E_act_ion"])
        self.t_elec = float(params["thickness_electrolyte"])
        self.i0_a_ref = float(params["i0_anode_ref"])
        self.E_act_a = float(params["E_act_anode"])
        self.i0_c_ref = float(params["i0_cathode_ref"])
        self.E_act_c = float(params["E_act_cathode"])
        self.alpha = float(params["alpha"])
        self.fuel_util = float(params.get("fuel_utilization", 0.7))

    # ------------------------------------------------------------------
    # Nernst voltage
    # ------------------------------------------------------------------

    def nernst_voltage(self, T):
        """Nernst OCV [V]. Uses Gibbs free energy linear approximation."""
        T = np.asarray(T, dtype=float)
        # Standard potential varies with T: E0(T) ~ 1.253 - 0.00024*(T-298)
        E0_T = 1.253 - 0.00024 * (T - 298.15)
        return E0_T + (self.R * T) / (2.0 * self.F) * np.log(
            self.pH2 * np.sqrt(self.pO2) / self.pH2O
        )

    # ------------------------------------------------------------------
    # YSZ ionic conductivity
    # ------------------------------------------------------------------

    def ionic_conductivity(self, T):
        """YSZ electrolyte ionic conductivity [S/cm]."""
        T = np.asarray(T, dtype=float)
        return (self.A_sigma / T) * np.exp(-self.E_act_ion / (self.R * T))

    def ohmic_asr(self, T):
        """Ohmic area-specific resistance [ohm cm2]."""
        sigma = self.ionic_conductivity(T)
        return self.t_elec / sigma

    # ------------------------------------------------------------------
    # Exchange current densities (Arrhenius)
    # ------------------------------------------------------------------

    def i0_anode(self, T):
        T = np.asarray(T, dtype=float)
        return self.i0_a_ref * np.exp(
            -self.E_act_a / self.R * (1.0 / T - 1.0 / self.T_ref)
        )

    def i0_cathode(self, T):
        T = np.asarray(T, dtype=float)
        return self.i0_c_ref * np.exp(
            -self.E_act_c / self.R * (1.0 / T - 1.0 / self.T_ref)
        )

    # ------------------------------------------------------------------
    # Loss terms
    # ------------------------------------------------------------------

    def activation_loss(self, j, T):
        """Combined anode + cathode activation overpotential [V]."""
        j = np.asarray(j, dtype=float)
        T = np.asarray(T, dtype=float)
        j_safe = np.maximum(j, 1e-10)
        i0_a = self.i0_anode(T)
        i0_c = self.i0_cathode(T)
        V_a = (self.R * T) / (self.alpha * self.n * self.F) * np.arcsinh(j_safe / (2.0 * i0_a))
        V_c = (self.R * T) / (self.alpha * self.n * self.F) * np.arcsinh(j_safe / (2.0 * i0_c))
        return V_a + V_c

    def ohmic_loss(self, j, T):
        """Ohmic voltage loss [V]."""
        j = np.asarray(j, dtype=float)
        return j * self.ohmic_asr(T)

    def concentration_loss(self, j, T):
        """Concentration loss [V]."""
        j = np.asarray(j, dtype=float)
        T = np.asarray(T, dtype=float)
        ratio = np.minimum(j / self.j_L, 0.9999)
        return np.where(
            j > 0,
            -(self.R * T) / (self.n * self.F) * np.log(1.0 - ratio),
            0.0,
        )

    # ------------------------------------------------------------------
    # Cell voltage and derived
    # ------------------------------------------------------------------

    def cell_voltage(self, j, T):
        """Cell voltage [V]."""
        j = np.asarray(j, dtype=float)
        T = np.asarray(T, dtype=float)
        E = self.nernst_voltage(T)
        V = E - self.activation_loss(j, T) - self.ohmic_loss(j, T) - self.concentration_loss(j, T)
        return np.clip(V, 0.0, None)

    def power_density(self, j, T):
        return np.asarray(j, dtype=float) * self.cell_voltage(j, T)

    def efficiency(self, j, T):
        """Voltage efficiency based on LHV."""
        V = self.cell_voltage(j, T)
        E_LHV = self.LHV_H2 / (self.n * self.F)  # ~1.253 V
        return np.where(j > 0, V / E_LHV, 0.0)

    def asr_total(self, j, T):
        """Total effective ASR [ohm cm2] = (E_nernst - V_cell) / j."""
        j = np.asarray(j, dtype=float)
        E = self.nernst_voltage(T)
        V = self.cell_voltage(j, T)
        j_safe = np.maximum(j, 1e-10)
        return (E - V) / j_safe

    def heat_generation(self, j, T):
        """Heat generation [W/cm2]."""
        j = np.asarray(j, dtype=float)
        V = self.cell_voltage(j, T)
        return j * (self.E_tn - V)

    def evaluate(self, j, T):
        j = np.asarray(j, dtype=float)
        T = np.asarray(T, dtype=float)
        E = self.nernst_voltage(T)
        V_act = self.activation_loss(j, T)
        V_ohm_val = self.ohmic_loss(j, T)
        V_conc = self.concentration_loss(j, T)
        V_cell = np.clip(E - V_act - V_ohm_val - V_conc, 0.0, None)

        return {
            "cell_voltage": V_cell,
            "power_density": j * V_cell,
            "efficiency": self.efficiency(j, T),
            "asr": self.asr_total(j, T),
            "heat_generation": j * (self.E_tn - V_cell),
            "E_nernst": E,
            "V_act": V_act,
            "V_ohm": V_ohm_val,
            "V_conc": V_conc,
            "ohmic_asr": self.ohmic_asr(T),
        }
