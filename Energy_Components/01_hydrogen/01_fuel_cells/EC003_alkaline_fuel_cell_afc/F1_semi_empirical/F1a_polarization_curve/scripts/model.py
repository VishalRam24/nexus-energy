"""
EC003 -- Alkaline Fuel Cell (AFC) -- F1a Polarization Curve
Basic polarization: V = E_rev - V_act - V_ohm - V_conc

AFC operates at 60-80 C with KOH electrolyte (6 mol/L typical).
E_rev for KOH electrolyte ~1.0 V at operating temperature.
Tafel kinetics for activation; linear ohmic; log concentration.

References:
    Larminie & Dicks (2003) Fuel Cell Systems Explained, 2nd Ed., Wiley.
    Appleby & Foulkes (1989) Fuel Cell Handbook, Van Nostrand Reinhold.
    Gilliam et al. (2007) Int. J. Hydrogen Energy, 32(3), 359-364.
"""

import numpy as np


class AFCModel:
    """
    Semi-empirical single-cell polarization curve for an alkaline fuel cell.

    V_cell = E_rev(T) - V_act(j,T) - V_ohm(j) - V_conc(j)

    E_rev  = 1.229 - 0.000846*(T - 298.15) + RT/(2F)*ln(pH2*sqrt(pO2))
    V_act  = RT/(alpha*n*F) * arcsinh(j / (2*i0))      [Tafel/BV]
    V_ohm  = j * R_ohm                                  [area-specific]
    V_conc = -B * ln(1 - j/j_L)
    """

    R = 8.314
    F = 96485.0
    n = 2
    E_tn = 1.481  # thermoneutral voltage (HHV) [V]

    def __init__(self, params: dict):
        self.T         = float(params["T"])
        self.N_cells   = int(params["N_cells"])
        self.A_cell    = float(params["A_cell"])
        self.pH2       = float(params["pH2"])
        self.pO2       = float(params["pO2"])
        self.j_L       = float(params["j_L"])
        self.i0        = float(params["i0"])
        self.alpha     = float(params.get("alpha", 0.5))
        self.R_ohm     = float(params["R_ohm"])
        self.B_conc    = float(params.get("B_conc", 0.010))

    def nernst_voltage(self, T=None):
        T = T if T is not None else self.T
        return (
            1.229
            - 0.000846 * (T - 298.15)
            + (self.R * T) / (2.0 * self.F) * np.log(self.pH2 * np.sqrt(self.pO2))
        )

    def activation_loss(self, j, T=None):
        T = T if T is not None else self.T
        j_safe = max(j, 1e-10)
        return (self.R * T) / (self.alpha * self.n * self.F) * np.arcsinh(
            j_safe / (2.0 * self.i0)
        )

    def ohmic_loss(self, j):
        return j * self.R_ohm

    def concentration_loss(self, j):
        if j <= 0:
            return 0.0
        ratio = j / self.j_L
        if ratio >= 1.0:
            return float("inf")
        return -self.B_conc * np.log(1.0 - ratio)

    def cell_voltage(self, j, T=None):
        T = T if T is not None else self.T
        E = self.nernst_voltage(T)
        V_act = self.activation_loss(j, T)
        V_ohm = self.ohmic_loss(j)
        V_conc = self.concentration_loss(j)
        return max(0.0, E - V_act - V_ohm - V_conc)

    def stack_voltage(self, j, T=None):
        return self.N_cells * self.cell_voltage(j, T)

    def power_density(self, j, T=None):
        return j * self.cell_voltage(j, T)

    def stack_power(self, j, T=None):
        return self.N_cells * self.power_density(j, T) * self.A_cell

    def efficiency(self, j, T=None):
        V = self.cell_voltage(j, T)
        return max(0.0, V / self.E_tn)

    def evaluate(self, j, T_celsius=None):
        if j < 0:
            raise ValueError(f"Current density must be >= 0, got {j}")
        if j >= self.j_L:
            raise ValueError(f"Current density {j} >= limiting current density {self.j_L}")

        T_K = (T_celsius + 273.15) if T_celsius is not None else self.T

        E = self.nernst_voltage(T_K)
        V_act = self.activation_loss(j, T_K)
        V_ohm = self.ohmic_loss(j)
        V_conc = self.concentration_loss(j)
        V_cell = max(0.0, E - V_act - V_ohm - V_conc)
        V_stack = self.N_cells * V_cell
        P_density = j * V_cell
        P_stack = self.N_cells * P_density * self.A_cell
        eta = max(0.0, V_cell / self.E_tn)

        return {
            "E_Nernst_V":          E,
            "V_act_V":             V_act,
            "V_ohm_V":             V_ohm,
            "V_conc_V":            V_conc,
            "cell_voltage_V":      V_cell,
            "stack_voltage_V":     V_stack,
            "power_density_W_cm2": P_density,
            "stack_power_W":       P_stack,
            "efficiency":          eta,
        }
