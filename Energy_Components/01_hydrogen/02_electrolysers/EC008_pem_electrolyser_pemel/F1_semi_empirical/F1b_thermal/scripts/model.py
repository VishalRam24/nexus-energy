"""
EC008 -- PEM Electrolyser (PEMEL) -- F1b Thermal Model
Temperature-dependent V-I characteristic with Arrhenius kinetics.

Physics:
  - Thermoneutral voltage: E_tn(T) = 1.481 - 0.000223*(T - 298)
  - Reversible voltage: E_rev(T) = 1.229 - 0.000846*(T - 298.15)
  - Anode activation: V_act_a = RT/(alpha_a*2*F)*arcsinh(j/(2*i0_a(T)))
  - Cathode activation: V_act_c = RT/(alpha_c*2*F)*arcsinh(j/(2*i0_c(T)))
  - Exchange current: i0(T) = i0_ref * exp(-E_act/R*(1/T - 1/T_ref))
  - Membrane: sigma(T) = (0.005139*lambda - 0.00326)*exp(1268*(1/303 - 1/T))
  - V_cell = E_rev + V_act_a + V_act_c + V_ohm
  - Heat: Q = j*(V_cell - E_tn), positive when V_cell > E_tn (exothermic)

References:
    Garcia-Valverde et al. (2012), Int. J. Hydrogen Energy, 37(2), 1927-1938
    Springer et al. (1991), J. Electrochem. Soc., 138(8), 2334-2342
"""

import numpy as np


class PEMELThermalModel:
    """
    PEM electrolyser with explicit temperature dependence.
    Operating range: 323-363 K (50-90 C).
    """

    R = 8.314
    F = 96485.0
    n = 2
    HHV_H2 = 286000.0  # J/mol

    def __init__(self, params: dict):
        self.T_ref = float(params["T_ref"])
        self.N_cells = int(params["N_cells"])
        self.A_cell = float(params["A_cell"])
        self.i0_a_ref = float(params["i0_anode_ref"])
        self.E_act_a = float(params["E_act_anode"])
        self.i0_c_ref = float(params["i0_cathode_ref"])
        self.E_act_c = float(params["E_act_cathode"])
        self.alpha_a = float(params.get("alpha_a", 0.5))
        self.alpha_c = float(params.get("alpha_c", 0.5))
        self.sigma_ref = float(params["sigma_ref"])
        self.t_mem = float(params["membrane_thickness"])
        self.lambda_mem = float(params.get("lambda_mem", 14.0))
        self.pressure = float(params.get("pressure", 1.0))
        self.eta_F = float(params.get("faradaic_efficiency", 0.99))

    # ------------------------------------------------------------------
    # Thermodynamic voltages
    # ------------------------------------------------------------------

    def thermoneutral_voltage(self, T):
        """Thermoneutral voltage [V] — enthalpy-based."""
        T = np.asarray(T, dtype=float)
        return 1.481 - 0.000223 * (T - 298.15)

    def reversible_voltage(self, T):
        """Reversible (Nernst) voltage [V]."""
        T = np.asarray(T, dtype=float)
        return 1.229 - 0.000846 * (T - 298.15)

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
    # Activation overpotentials
    # ------------------------------------------------------------------

    def activation_anode(self, j, T):
        j = np.asarray(j, dtype=float)
        T = np.asarray(T, dtype=float)
        j_safe = np.maximum(j, 1e-10)
        i0 = self.i0_anode(T)
        return (self.R * T) / (self.alpha_a * self.n * self.F) * np.arcsinh(
            j_safe / (2.0 * i0)
        )

    def activation_cathode(self, j, T):
        j = np.asarray(j, dtype=float)
        T = np.asarray(T, dtype=float)
        j_safe = np.maximum(j, 1e-10)
        i0 = self.i0_cathode(T)
        return (self.R * T) / (self.alpha_c * self.n * self.F) * np.arcsinh(
            j_safe / (2.0 * i0)
        )

    # ------------------------------------------------------------------
    # Membrane resistance (Springer model)
    # ------------------------------------------------------------------

    def membrane_conductivity(self, T):
        T = np.asarray(T, dtype=float)
        sigma_303 = 0.005139 * self.lambda_mem - 0.00326
        return sigma_303 * np.exp(1268.0 * (1.0 / 303.0 - 1.0 / T))

    def membrane_resistance(self, T):
        """Area-specific resistance [ohm cm2]."""
        return self.t_mem / self.membrane_conductivity(T)

    def ohmic_loss(self, j, T):
        j = np.asarray(j, dtype=float)
        return j * self.membrane_resistance(T)

    # ------------------------------------------------------------------
    # Cell voltage (electrolyser: V = E_rev + losses)
    # ------------------------------------------------------------------

    def cell_voltage(self, j, T):
        """Cell voltage [V] — for electrolyser, V > E_rev."""
        j = np.asarray(j, dtype=float)
        T = np.asarray(T, dtype=float)
        E_rev = self.reversible_voltage(T)
        V_act_a = self.activation_anode(j, T)
        V_act_c = self.activation_cathode(j, T)
        V_ohm = self.ohmic_loss(j, T)
        return E_rev + V_act_a + V_act_c + V_ohm

    # ------------------------------------------------------------------
    # Derived quantities
    # ------------------------------------------------------------------

    def power_consumption(self, j, T):
        """Power consumption per unit area [W/cm2]."""
        return np.asarray(j, dtype=float) * self.cell_voltage(j, T)

    def efficiency_voltage(self, j, T):
        """Voltage efficiency: E_tn / V_cell."""
        V = self.cell_voltage(j, T)
        E_tn = self.thermoneutral_voltage(T)
        return np.where(V > 0, E_tn / V, 0.0)

    def efficiency_faradaic(self, j, T):
        """Faradaic efficiency (constant or model-based)."""
        return np.full_like(np.asarray(j, dtype=float), self.eta_F)

    def h2_production_rate(self, j, T):
        """H2 production rate [mol/s/cm2] via Faraday's law."""
        j = np.asarray(j, dtype=float)
        return self.eta_F * j / (self.n * self.F)

    def heat_generation(self, j, T):
        """Heat generation [W/cm2]. Positive when V > E_tn (exothermic waste heat)."""
        j = np.asarray(j, dtype=float)
        V = self.cell_voltage(j, T)
        E_tn = self.thermoneutral_voltage(T)
        return j * (V - E_tn)

    def evaluate(self, j, T):
        j = np.asarray(j, dtype=float)
        T = np.asarray(T, dtype=float)

        E_rev = self.reversible_voltage(T)
        E_tn = self.thermoneutral_voltage(T)
        V_act_a = self.activation_anode(j, T)
        V_act_c = self.activation_cathode(j, T)
        V_ohm = self.ohmic_loss(j, T)
        V_cell = E_rev + V_act_a + V_act_c + V_ohm

        return {
            "cell_voltage": V_cell,
            "power_consumption": j * V_cell,
            "efficiency_voltage": np.where(V_cell > 0, E_tn / V_cell, 0.0),
            "efficiency_faradaic": np.full_like(j, self.eta_F),
            "h2_production_rate": self.eta_F * j / (self.n * self.F),
            "heat_generation": j * (V_cell - E_tn),
            "E_rev": E_rev,
            "E_tn": E_tn,
            "V_act_anode": V_act_a,
            "V_act_cathode": V_act_c,
            "V_ohm": V_ohm,
            "membrane_resistance": self.membrane_resistance(T),
        }
