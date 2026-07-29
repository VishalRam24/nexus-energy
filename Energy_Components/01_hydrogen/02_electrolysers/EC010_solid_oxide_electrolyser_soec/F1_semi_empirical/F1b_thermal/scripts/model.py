"""
EC010 -- Solid Oxide Electrolyser (SOEC) -- F1b Thermal Model
Temperature-dependent V-I with YSZ conductivity and Arrhenius electrode kinetics.

Physics:
  - E_rev(T) = 1.253 - 0.00024*(T - 298.15)  [reversible voltage for steam electrolysis]
  - E_tn(T) = 1.285 - 0.000065*(T - 298.15)   [thermoneutral voltage]
  - YSZ: sigma_ion(T) = (A/T)*exp(-E_act/(RT))
  - Ohmic: R_ohm = t_elec / sigma_ion(T)
  - Activation (Arrhenius + Butler-Volmer):
      i0_a(T) = i0_a_ref * exp(-E_act_a/R*(1/T - 1/T_ref))
      i0_c(T) = i0_c_ref * exp(-E_act_c/R*(1/T - 1/T_ref))
      V_act = RT/(alpha*n*F)*[arcsinh(j/(2*i0_a)) + arcsinh(j/(2*i0_c))]
  - V_cell = E_rev + V_act + V_ohm
  - Thermal mode: endothermic (V < E_tn), thermoneutral (V ~ E_tn), exothermic (V > E_tn)

References:
    Ni et al. (2007), Chem. Eng. Tech., 29(6), 636-642
    Udagawa et al. (2007), J. Power Sources, 166(1), 127-136
    Kazempoor & Braun (2014), Int. J. Hydrogen Energy, 39(5), 2669-2684
"""

import numpy as np


class SOECThermalModel:
    """
    Solid oxide electrolyser with explicit temperature dependence.
    Operating range: 973-1123 K (700-850 C).
    """

    R = 8.314
    F = 96485.0
    n = 2
    LHV_H2 = 241800.0  # J/mol

    def __init__(self, params: dict):
        self.T_ref = float(params["T_ref"])
        self.N_cells = int(params["N_cells"])
        self.A_cell = float(params["A_cell"])
        self.A_sigma = float(params["A_sigma"])
        self.E_act_ion = float(params["E_act_ion"])
        self.t_elec = float(params["thickness_electrolyte"])
        self.i0_a_ref = float(params["i0_anode_ref"])
        self.E_act_a = float(params["E_act_anode"])
        self.i0_c_ref = float(params["i0_cathode_ref"])
        self.E_act_c = float(params["E_act_cathode"])
        self.alpha = float(params["alpha"])
        self.j_L = float(params["j_L"])
        self.steam_util = float(params.get("steam_utilization", 0.6))

    # ------------------------------------------------------------------
    # Thermodynamic voltages
    # ------------------------------------------------------------------

    def reversible_voltage(self, T):
        """Reversible voltage for steam electrolysis [V]."""
        T = np.asarray(T, dtype=float)
        return 1.253 - 0.00024 * (T - 298.15)

    def thermoneutral_voltage(self, T):
        """Thermoneutral voltage [V] — enthalpy-based."""
        T = np.asarray(T, dtype=float)
        return 1.285 - 0.000065 * (T - 298.15)

    # ------------------------------------------------------------------
    # YSZ ionic conductivity (same model as SOFC)
    # ------------------------------------------------------------------

    def ionic_conductivity(self, T):
        T = np.asarray(T, dtype=float)
        return (self.A_sigma / T) * np.exp(-self.E_act_ion / (self.R * T))

    def ohmic_asr(self, T):
        """Ohmic ASR from electrolyte [ohm cm2]."""
        return self.t_elec / self.ionic_conductivity(T)

    # ------------------------------------------------------------------
    # Exchange current densities
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
        """Combined anode + cathode activation [V]."""
        j = np.asarray(j, dtype=float)
        T = np.asarray(T, dtype=float)
        j_safe = np.maximum(j, 1e-10)
        i0_a = self.i0_anode(T)
        i0_c = self.i0_cathode(T)
        V_a = (self.R * T) / (self.alpha * self.n * self.F) * np.arcsinh(j_safe / (2.0 * i0_a))
        V_c = (self.R * T) / (self.alpha * self.n * self.F) * np.arcsinh(j_safe / (2.0 * i0_c))
        return V_a + V_c

    def ohmic_loss(self, j, T):
        j = np.asarray(j, dtype=float)
        return j * self.ohmic_asr(T)

    # ------------------------------------------------------------------
    # Cell voltage (electrolyser: V = E_rev + losses)
    # ------------------------------------------------------------------

    def cell_voltage(self, j, T):
        """Cell voltage [V]."""
        j = np.asarray(j, dtype=float)
        T = np.asarray(T, dtype=float)
        E_rev = self.reversible_voltage(T)
        V_act = self.activation_loss(j, T)
        V_ohm = self.ohmic_loss(j, T)
        return E_rev + V_act + V_ohm

    # ------------------------------------------------------------------
    # Derived quantities
    # ------------------------------------------------------------------

    def power_consumption(self, j, T):
        """Power consumption per unit area [W/cm2]."""
        return np.asarray(j, dtype=float) * self.cell_voltage(j, T)

    def efficiency(self, j, T):
        """Efficiency = E_tn / V_cell (thermoneutral basis)."""
        V = self.cell_voltage(j, T)
        E_tn = self.thermoneutral_voltage(T)
        return np.where(V > 0, E_tn / V, 0.0)

    def h2_production_rate(self, j):
        """H2 production rate [mol/s/cm2] via Faraday's law (100% Faradaic)."""
        j = np.asarray(j, dtype=float)
        return j / (self.n * self.F)

    def thermal_mode(self, j, T):
        """
        Thermal operation mode.
        Returns string: 'endothermic', 'thermoneutral', or 'exothermic'.
        For arrays, returns numeric: +1=endothermic, 0=thermoneutral, -1=exothermic.
        """
        V = self.cell_voltage(j, T)
        E_tn = self.thermoneutral_voltage(T)
        diff = E_tn - V
        return np.where(diff > 0.005, 1, np.where(diff < -0.005, -1, 0))

    def heat_generation(self, j, T):
        """
        Heat generation [W/cm2].
        Q = j*(V_cell - E_tn).
        Positive = exothermic (waste heat), negative = endothermic (needs heat input).
        """
        j = np.asarray(j, dtype=float)
        V = self.cell_voltage(j, T)
        E_tn = self.thermoneutral_voltage(T)
        return j * (V - E_tn)

    def evaluate(self, j, T):
        j = np.asarray(j, dtype=float)
        T = np.asarray(T, dtype=float)

        E_rev = self.reversible_voltage(T)
        E_tn = self.thermoneutral_voltage(T)
        V_act = self.activation_loss(j, T)
        V_ohm = self.ohmic_loss(j, T)
        V_cell = E_rev + V_act + V_ohm

        return {
            "cell_voltage": V_cell,
            "power_consumption": j * V_cell,
            "efficiency": np.where(V_cell > 0, E_tn / V_cell, 0.0),
            "h2_production_rate": j / (self.n * self.F),
            "thermal_mode": self.thermal_mode(j, T),
            "heat_generation": j * (V_cell - E_tn),
            "E_rev": E_rev,
            "E_tn": E_tn,
            "V_act": V_act,
            "V_ohm": V_ohm,
            "ohmic_asr": self.ohmic_asr(T),
        }
