"""
EC008 — PEM Electrolyser (PEMEL) — F1a V-I Characteristic
Physics equations class.

Model: V_cell = E_rev + V_act + V_ohm
Source: Garcia-Valverde et al. (2012), Int. J. Hydrogen Energy, 37(2), 1927-1938
"""

import numpy as np


class PEMElectrolyserModel:
    """
    Semi-empirical single-cell V-I model for a PEM electrolyser.

    Governing equations
    -------------------
    E_rev   = 1.229 - 0.0009*(T - 298)                            [V]
    V_act   = (R*T) / (alpha*n*F) * arcsinh(j / (2*j0))           [V]
    V_ohm   = j * R_membrane                                       [V]
    V_cell  = E_rev + V_act + V_ohm                                [V]
    V_stack = N_cells * V_cell                                     [V]

    Hydrogen production rate via Faraday's law:
    n_H2 = (j * A_electrode) / (2 * F)                            [mol/s]

    Efficiency (HHV basis):
    eta = (n_H2 * HHV_H2) / P_stack
    """

    # Physical constants
    R = 8.314       # J/(mol·K)  — universal gas constant
    F = 96485.0     # C/mol      — Faraday constant
    n = 2           # electrons transferred per H2 molecule
    HHV_H2 = 286e3  # J/mol      — higher heating value of H2

    def __init__(self, params: dict):
        """
        Parameters
        ----------
        params : dict
            T               : operating temperature [K]
            N_cells         : number of cells in stack
            electrode_area  : active electrode area [cm²]
            j0              : exchange current density [A/cm²]
            alpha           : charge transfer coefficient [-]
            R_membrane      : membrane area resistance [Ω·cm²]
        """
        self.T = float(params["T"])
        self.N_cells = int(params["N_cells"])
        self.electrode_area = float(params["electrode_area"])
        self.j0 = float(params["j0"])
        self.alpha = float(params["alpha"])
        self.R_membrane = float(params["R_membrane"])

    # ------------------------------------------------------------------
    # Individual voltage contributions
    # ------------------------------------------------------------------

    def reversible_voltage(self, T: float = None) -> float:
        """Simplified Nernst/thermodynamic open-circuit voltage [V]."""
        T = T if T is not None else self.T
        return 1.229 - 0.0009 * (T - 298.0)

    def activation_overpotential(self, j: float, T: float = None) -> float:
        """Butler-Volmer activation overpotential [V]."""
        T = T if T is not None else self.T
        # arcsinh formulation valid for both anode and cathode combined
        return (self.R * T) / (self.alpha * self.n * self.F) * np.arcsinh(j / (2.0 * self.j0))

    def ohmic_overpotential(self, j: float) -> float:
        """Ohmic loss across Nafion membrane [V]."""
        return j * self.R_membrane

    # ------------------------------------------------------------------
    # Cell and stack voltage
    # ------------------------------------------------------------------

    def cell_voltage(self, j: float, T: float = None) -> float:
        """Total cell voltage [V]."""
        T = T if T is not None else self.T
        E_rev = self.reversible_voltage(T)
        V_act = self.activation_overpotential(j, T)
        V_ohm = self.ohmic_overpotential(j)
        return E_rev + V_act + V_ohm

    def stack_voltage(self, j: float, T: float = None) -> float:
        """Stack voltage for N_cells [V]."""
        return self.N_cells * self.cell_voltage(j, T)

    # ------------------------------------------------------------------
    # Derived quantities
    # ------------------------------------------------------------------

    def hydrogen_production_rate(self, j: float) -> float:
        """
        Molar hydrogen production rate via Faraday's law [mol/s].
        Assumes 100 % Faradaic efficiency (conservative upper bound).
        """
        I_total = j * self.electrode_area   # total current [A]
        return I_total / (self.n * self.F)

    def stack_power(self, j: float, T: float = None) -> float:
        """Electrical power input to the stack [W]."""
        I_total = j * self.electrode_area
        return self.stack_voltage(j, T) * I_total

    def efficiency(self, j: float, T: float = None) -> float:
        """
        Stack efficiency on HHV basis [-].
        eta = (n_H2 * HHV_H2) / P_stack
        """
        if j <= 0.0:
            return 0.0
        P = self.stack_power(j, T)
        n_H2 = self.hydrogen_production_rate(j)
        return (n_H2 * self.HHV_H2) / P

    # ------------------------------------------------------------------
    # Full operating-point evaluation
    # ------------------------------------------------------------------

    def evaluate(self, j: float, T_celsius: float = None) -> dict:
        """
        Return all outputs for a given current density and temperature.

        Parameters
        ----------
        j         : current density [A/cm²]
        T_celsius : temperature [°C]; uses default T if None

        Returns
        -------
        dict with keys: j, T_K, E_rev, V_act, V_ohm, cell_voltage,
                        stack_voltage, hydrogen_rate_mol_s,
                        power_W, efficiency
        """
        if j < 0:
            raise ValueError(f"Current density must be >= 0, got {j}")

        T_K = (T_celsius + 273.15) if T_celsius is not None else self.T

        E_rev = self.reversible_voltage(T_K)
        V_act = self.activation_overpotential(j, T_K)
        V_ohm = self.ohmic_overpotential(j)
        V_cell = E_rev + V_act + V_ohm
        V_stack = self.N_cells * V_cell
        I_total = j * self.electrode_area
        n_H2 = self.hydrogen_production_rate(j)
        P = V_stack * I_total
        eta = self.efficiency(j, T_K)

        return {
            "j_A_cm2": j,
            "T_K": T_K,
            "E_rev_V": E_rev,
            "V_act_V": V_act,
            "V_ohm_V": V_ohm,
            "cell_voltage_V": V_cell,
            "stack_voltage_V": V_stack,
            "hydrogen_rate_mol_s": n_H2,
            "power_W": P,
            "efficiency": eta,
        }
