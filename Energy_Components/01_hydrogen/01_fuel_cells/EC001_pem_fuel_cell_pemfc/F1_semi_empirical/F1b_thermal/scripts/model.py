"""
EC001 -- PEM Fuel Cell (PEMFC) -- F1b Thermal Model
Temperature-dependent polarization curve with Arrhenius kinetics.

Extends F1a by making all loss mechanisms explicitly temperature-dependent:
  - Nernst potential: E_rev(T) = 1.229 - 0.000846*(T - 298.15) + RT/(2F)*ln(pH2*sqrt(pO2))
  - Exchange current density: i0(T) = i0_ref * exp(-E_act/R * (1/T - 1/T_ref))
  - Activation loss: V_act = RT/(alpha*n*F) * arcsinh(j / (2*i0(T)))
  - Membrane conductivity: sigma(T) = sigma_ref * exp(1268*(1/303 - 1/T))
  - Ohmic loss: V_ohm = j * t_mem / sigma(T)
  - Concentration loss: V_conc = -B * ln(1 - j/j_L)
  - Heat generation: Q = j * (E_tn - V_cell) where E_tn = 1.481 V (thermoneutral)

References:
    Amphlett et al. (1995), J. Electrochem. Soc., 142(1), 1-8
    Springer et al. (1991), J. Electrochem. Soc., 138(8), 2334-2342
"""

import numpy as np


class PEMFCThermalModel:
    """
    PEM fuel cell with explicit temperature dependence on all loss terms.

    Operating range: 333-363 K (60-90 C).
    """

    # Physical constants
    R = 8.314       # J/(mol K)
    F = 96485.0     # C/mol
    n = 2           # electrons per H2 molecule
    E_tn = 1.481    # thermoneutral voltage (HHV basis) [V]

    def __init__(self, params: dict):
        self.T_ref = float(params["T_ref"])
        self.N_cells = int(params["N_cells"])
        self.A_cell = float(params["A_cell"])
        self.pH2 = float(params["pH2"])
        self.pO2 = float(params["pO2"])
        self.j_L = float(params["j_L"])
        self.i0_ref = float(params["i0_ref"])
        self.E_act = float(params["E_act"])
        self.sigma_ref = float(params["sigma_ref"])
        self.t_mem = float(params["membrane_thickness"])
        self.lambda_mem = float(params.get("lambda_mem", 14.0))
        self.alpha = float(params.get("alpha", 0.5))
        self.B_conc = float(params.get("B_conc", 0.016))

    # ------------------------------------------------------------------
    # Nernst (reversible) voltage
    # ------------------------------------------------------------------

    def nernst_voltage(self, T):
        """Open-circuit (Nernst) voltage [V]."""
        T = np.asarray(T, dtype=float)
        return (
            1.229
            - 0.000846 * (T - 298.15)
            + (self.R * T) / (2.0 * self.F) * np.log(self.pH2 * np.sqrt(self.pO2))
        )

    # ------------------------------------------------------------------
    # Exchange current density (Arrhenius)
    # ------------------------------------------------------------------

    def exchange_current_density(self, T):
        """Temperature-dependent exchange current density [A/cm2]."""
        T = np.asarray(T, dtype=float)
        return self.i0_ref * np.exp(
            -self.E_act / self.R * (1.0 / T - 1.0 / self.T_ref)
        )

    # ------------------------------------------------------------------
    # Activation loss (Butler-Volmer arcsinh form)
    # ------------------------------------------------------------------

    def activation_loss(self, j, T):
        """Activation overpotential [V]."""
        j = np.asarray(j, dtype=float)
        T = np.asarray(T, dtype=float)
        i0_T = self.exchange_current_density(T)
        # Avoid log(0) at j=0
        j_safe = np.maximum(j, 1e-10)
        return (self.R * T) / (self.alpha * self.n * self.F) * np.arcsinh(
            j_safe / (2.0 * i0_T)
        )

    # ------------------------------------------------------------------
    # Membrane conductivity and ohmic loss
    # ------------------------------------------------------------------

    def membrane_conductivity(self, T):
        """
        Nafion membrane conductivity [S/cm].
        Springer et al. (1991) model with Arrhenius temperature dependence.
        """
        T = np.asarray(T, dtype=float)
        lam = self.lambda_mem
        sigma_303 = 0.005139 * lam - 0.00326
        return sigma_303 * np.exp(1268.0 * (1.0 / 303.0 - 1.0 / T))

    def membrane_resistance(self, T):
        """Membrane area-specific resistance [ohm cm2]."""
        sigma = self.membrane_conductivity(T)
        return self.t_mem / sigma

    def ohmic_loss(self, j, T):
        """Ohmic voltage loss [V]."""
        j = np.asarray(j, dtype=float)
        return j * self.membrane_resistance(T)

    # ------------------------------------------------------------------
    # Concentration loss
    # ------------------------------------------------------------------

    def concentration_loss(self, j):
        """Concentration (mass transport) loss [V]."""
        j = np.asarray(j, dtype=float)
        ratio = j / self.j_L
        ratio_safe = np.minimum(ratio, 0.9999)
        return np.where(
            j > 0,
            -self.B_conc * np.log(1.0 - ratio_safe),
            0.0,
        )

    # ------------------------------------------------------------------
    # Cell voltage
    # ------------------------------------------------------------------

    def cell_voltage(self, j, T):
        """Net cell voltage [V]."""
        j = np.asarray(j, dtype=float)
        T = np.asarray(T, dtype=float)
        E = self.nernst_voltage(T)
        V_act = self.activation_loss(j, T)
        V_ohm = self.ohmic_loss(j, T)
        V_conc = self.concentration_loss(j)
        V = E - V_act - V_ohm - V_conc
        return np.clip(V, 0.0, None)

    # ------------------------------------------------------------------
    # Derived quantities
    # ------------------------------------------------------------------

    def power_density(self, j, T):
        """Power density [W/cm2]."""
        return np.asarray(j, dtype=float) * self.cell_voltage(j, T)

    def efficiency(self, j, T):
        """Voltage efficiency relative to HHV (1.481 V) [-]."""
        V = self.cell_voltage(j, T)
        return V / self.E_tn

    def heat_generation(self, j, T):
        """
        Heat generation per unit area [W/cm2].
        Q = j * (E_tn - V_cell).  Always >= 0 for fuel cells.
        """
        j = np.asarray(j, dtype=float)
        V = self.cell_voltage(j, T)
        return j * (self.E_tn - V)

    def evaluate(self, j, T):
        """
        Full operating-point evaluation.

        Parameters
        ----------
        j : float or array — current density [A/cm2]
        T : float or array — temperature [K]

        Returns
        -------
        dict with all outputs
        """
        j = np.asarray(j, dtype=float)
        T = np.asarray(T, dtype=float)

        E = self.nernst_voltage(T)
        V_act = self.activation_loss(j, T)
        V_ohm = self.ohmic_loss(j, T)
        V_conc = self.concentration_loss(j)
        V_cell = np.clip(E - V_act - V_ohm - V_conc, 0.0, None)
        P_density = j * V_cell
        eta = V_cell / self.E_tn
        Q = j * (self.E_tn - V_cell)
        R_mem = self.membrane_resistance(T)

        return {
            "cell_voltage": V_cell,
            "power_density": P_density,
            "efficiency": eta,
            "heat_generation": Q,
            "membrane_resistance": R_mem,
            "E_nernst": E,
            "V_act": V_act,
            "V_ohm": V_ohm,
            "V_conc": V_conc,
        }
