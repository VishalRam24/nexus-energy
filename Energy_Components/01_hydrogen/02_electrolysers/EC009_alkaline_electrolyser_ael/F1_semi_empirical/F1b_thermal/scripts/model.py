"""
EC009 -- Alkaline Electrolyser (AEL) -- F1b Thermal Model
Temperature-dependent V-I with KOH conductivity, bubble coverage, and Arrhenius kinetics.

Physics:
  - E_rev(T) = 1.229 - 0.000846*(T - 298.15)
  - KOH conductivity: sigma_KOH(T) = sigma_ref * exp(-E_act_koh/R * (1/T - 1/T_ref))
  - Bubble coverage: theta(j, T) = bubble_coeff * (j/j_L)^0.3 * (T_ref/T)^0.5
  - Effective conductivity: sigma_eff = sigma_KOH * (1 - theta)^1.5  (Bruggeman)
  - Ohmic: V_ohm = j * d_gap / (sigma_eff * 1e4)  [unit conversion m->cm]
  - Activation: V_act = RT/(alpha*n*F) * arcsinh(j_cm2/(2*i0(T)))
  - V_cell = E_rev + V_act + V_ohm

References:
    Ulleberg (2003), Int. J. Hydrogen Energy, 28(1), 21-33
    See & White (1997), J. Chem. Eng. Data, 42(6), 1266-1268
    Divisek et al. (1988), Electrochimica Acta, 33(11), 1515-1527
"""

import numpy as np


class AELThermalModel:
    """
    Alkaline electrolyser with explicit temperature and KOH concentration dependence.
    Operating range: 333-373 K, KOH 25-40 wt%.
    Current density in A/m2 (consistent with Ulleberg convention).
    """

    R = 8.314
    F = 96485.0
    n = 2
    HHV_H2 = 286000.0  # J/mol
    LHV_H2 = 241800.0  # J/mol

    def __init__(self, params: dict):
        self.T_ref = float(params["T_ref"])
        self.N_cells = int(params["N_cells"])
        self.A_cell = float(params["A_cell"])  # m2
        self.koh_conc = float(params.get("koh_concentration", 30.0))
        self.sigma_ref = float(params["sigma_KOH_ref"])
        self.E_act_koh = float(params["E_act_koh"])
        self.d_gap = float(params["electrode_gap"])  # m
        self.i0_ref = float(params["i0_ref"])  # A/cm2
        self.E_act_el = float(params["E_act_electrode"])
        self.alpha = float(params["alpha"])
        self.j_L = float(params["j_L"])  # A/m2
        self.bubble_coeff = float(params.get("bubble_coeff", 0.3))
        self.f1 = float(params.get("faradaic_f1", 250.0))
        self.f2 = float(params.get("faradaic_f2", 0.98))

    # ------------------------------------------------------------------
    # Reversible voltage
    # ------------------------------------------------------------------

    def reversible_voltage(self, T):
        T = np.asarray(T, dtype=float)
        return 1.229 - 0.000846 * (T - 298.15)

    # ------------------------------------------------------------------
    # KOH electrolyte conductivity
    # ------------------------------------------------------------------

    def koh_conductivity(self, T):
        """KOH electrolyte conductivity [S/cm] with Arrhenius T-dependence."""
        T = np.asarray(T, dtype=float)
        # Concentration correction factor (linear around 30 wt%)
        conc_factor = 1.0 + 0.02 * (self.koh_conc - 30.0)
        return self.sigma_ref * conc_factor * np.exp(
            -self.E_act_koh / self.R * (1.0 / T - 1.0 / self.T_ref)
        )

    def bubble_coverage(self, j, T):
        """Bubble coverage fraction theta(j, T)."""
        j = np.asarray(j, dtype=float)
        T = np.asarray(T, dtype=float)
        j_safe = np.maximum(j, 0.0)
        ratio = j_safe / self.j_L
        theta = self.bubble_coeff * np.power(ratio, 0.3) * np.sqrt(self.T_ref / T)
        return np.clip(theta, 0.0, 0.9)

    def effective_conductivity(self, j, T):
        """Effective electrolyte conductivity with bubble Bruggeman correction [S/cm]."""
        sigma = self.koh_conductivity(T)
        theta = self.bubble_coverage(j, T)
        return sigma * np.power(1.0 - theta, 1.5)

    # ------------------------------------------------------------------
    # Exchange current density
    # ------------------------------------------------------------------

    def exchange_current_density(self, T):
        """Temperature-dependent exchange current density [A/cm2]."""
        T = np.asarray(T, dtype=float)
        return self.i0_ref * np.exp(
            -self.E_act_el / self.R * (1.0 / T - 1.0 / self.T_ref)
        )

    # ------------------------------------------------------------------
    # Loss terms
    # ------------------------------------------------------------------

    def ohmic_loss(self, j, T):
        """Ohmic overpotential [V]. j in A/m2."""
        j = np.asarray(j, dtype=float)
        sigma_eff = self.effective_conductivity(j, T)  # S/cm
        # Convert gap from m to cm: d_gap [m] * 100 = [cm]
        d_cm = self.d_gap * 100.0
        # j in A/m2 -> A/cm2: j / 10000
        j_cm2 = j / 1e4
        return j_cm2 * d_cm / sigma_eff

    def activation_loss(self, j, T):
        """Activation overpotential [V]. j in A/m2."""
        j = np.asarray(j, dtype=float)
        T = np.asarray(T, dtype=float)
        j_cm2 = np.maximum(j / 1e4, 1e-10)
        i0 = self.exchange_current_density(T)
        return (self.R * T) / (self.alpha * self.n * self.F) * np.arcsinh(
            j_cm2 / (2.0 * i0)
        )

    # ------------------------------------------------------------------
    # Cell voltage
    # ------------------------------------------------------------------

    def cell_voltage(self, j, T):
        """Cell voltage [V]. j in A/m2."""
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
        """Power consumption per cell per unit area [W/m2]."""
        j = np.asarray(j, dtype=float)
        return j * self.cell_voltage(j, T)

    def faraday_efficiency(self, j):
        """Faradaic efficiency [-]."""
        j = np.asarray(j, dtype=float)
        I = j * self.A_cell  # A
        return np.where(j > 0,
                        np.clip(self.f1 * I**2 / (self.f2 + I**2), 0.0, 1.0),
                        0.0)

    def efficiency(self, j, T):
        """Overall efficiency = (H2 LHV) / (electrical input)."""
        j = np.asarray(j, dtype=float)
        V_cell = self.cell_voltage(j, T)
        eta_V = np.where(V_cell > 0, (self.LHV_H2 / (self.n * self.F)) / V_cell, 0.0)
        eta_F = self.faraday_efficiency(j)
        return np.clip(eta_V * eta_F, 0.0, 1.0)

    def h2_production_rate(self, j, T):
        """H2 production rate [mol/s] for the full stack."""
        j = np.asarray(j, dtype=float)
        I = j * self.A_cell
        eta_F = self.faraday_efficiency(j)
        return eta_F * self.N_cells * I / (2.0 * self.F)

    def evaluate(self, j, T):
        j = np.asarray(j, dtype=float)
        T = np.asarray(T, dtype=float)
        E_rev = self.reversible_voltage(T)
        V_act = self.activation_loss(j, T)
        V_ohm = self.ohmic_loss(j, T)
        V_cell = E_rev + V_act + V_ohm

        j_safe = np.maximum(j, 1e-10)
        I = j * self.A_cell
        P_stack = self.N_cells * V_cell * I  # W

        return {
            "cell_voltage": V_cell,
            "power_consumption": P_stack / 1000.0,  # kW
            "efficiency": self.efficiency(j, T),
            "h2_production_rate": self.h2_production_rate(j, T),
            "E_rev": E_rev,
            "V_act": V_act,
            "V_ohm": V_ohm,
            "koh_conductivity": self.koh_conductivity(T),
            "bubble_coverage": self.bubble_coverage(j, T),
        }
