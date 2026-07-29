"""
EC002 — Solid Oxide Fuel Cell (SOFC) — F1a Polarization Curve

V_cell = E_Nernst - V_act - V_ohm - V_conc

E_Nernst = E0 + (RT/2F) * ln(pH2 * sqrt(pO2) / pH2O)
V_act    = (RT / alpha*F) * arcsinh(j / (2*j0))       [Butler-Volmer simplified]
V_ohm    = j * ASR  (in consistent units: j [A/cm2], ASR [Ohm.cm2] -> V)
V_conc   = -(RT / nF) * ln(1 - j/j_L)
V_stack  = N_cells * V_cell

Stack power: P_stack = V_stack * I_stack
  where I_stack = j * A_cell  [A]

Efficiency: eta = (V_cell * n * F) / LHV_H2

Reference:
    Chan, S.H., Ho, H.K., Tian, Y. (2001), "Modelling of simple hybrid solid oxide
    fuel cell and gas turbine power plant", J. Power Sources, 93, 130-140.
"""

import numpy as np


class SOCFF1a:
    """Solid oxide fuel cell stack — Butler-Volmer + Ohmic + concentration polarization."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.T_op    = u["T_op"]["value"]      # K (default operating T)
        self.N       = u["N_cells"]["value"]
        self.ASR     = u["ASR"]["value"]       # Ohm.cm2
        self.j0      = u["j0"]["value"]        # A/cm2
        self.j_L     = u["j_L"]["value"]       # A/cm2
        self.alpha   = u["alpha"]["value"]
        self.pH2     = u["pH2"]["value"]       # atm
        self.pO2     = u["pO2"]["value"]       # atm
        self.pH2O    = u["pH2O"]["value"]      # atm
        self.E0      = u["E0"]["value"]        # V at 1073 K
        self.A_cell  = u["A_cell"]["value"]    # cm2
        self.LHV_H2  = u["LHV_H2"]["value"]   # J/mol
        self.F       = u["F"]["value"]         # C/mol
        self.R       = u["R"]["value"]         # J/molK
        self.n       = u["n"]["value"]

    def _temperature_K(self, temp_c):
        """Convert celsius input to Kelvin."""
        if temp_c is None:
            return self.T_op
        return np.asarray(temp_c, dtype=float) + 273.15

    def E_nernst(self, T_K):
        """Nernst open-circuit voltage [V]."""
        T = np.asarray(T_K, dtype=float)
        return self.E0 + (self.R * T / (2.0 * self.F)) * \
               np.log(self.pH2 * np.sqrt(self.pO2) / self.pH2O)

    def V_act(self, j, T_K):
        """Activation overpotential [V] — Butler-Volmer arcsinh form."""
        j = np.asarray(j, dtype=float)
        T = np.asarray(T_K, dtype=float)
        # j0 scales with temperature via Arrhenius; simple linear scale for F1
        # Use the provided j0 at T_op, scaled by (T/T_op)
        j0_T = self.j0 * (T / self.T_op)
        return (self.R * T / (self.alpha * self.F)) * np.arcsinh(j / (2.0 * j0_T))

    def V_ohm(self, j):
        """Ohmic overpotential [V] = j [A/cm2] * ASR [Ohm.cm2]."""
        return np.asarray(j, dtype=float) * self.ASR

    def V_conc(self, j, T_K):
        """Concentration overpotential [V]."""
        j  = np.asarray(j, dtype=float)
        T  = np.asarray(T_K, dtype=float)
        # Clamp j to avoid log(0) or negative argument
        j_safe = np.clip(j, 0.0, self.j_L * 0.9999)
        return -(self.R * T / (self.n * self.F)) * np.log(1.0 - j_safe / self.j_L)

    def cell_voltage(self, j, T_K):
        """Cell terminal voltage [V]."""
        j  = np.asarray(j, dtype=float)
        EN = self.E_nernst(T_K)
        VA = self.V_act(j, T_K)
        VO = self.V_ohm(j)
        VC = self.V_conc(j, T_K)
        V  = EN - VA - VO - VC
        return np.clip(V, 0.0, EN)   # voltage cannot exceed Nernst or go negative

    def predict(self, current_density, temperature_c=None):
        """
        Parameters
        ----------
        current_density : float or array [A/cm2]
        temperature_c   : float or array [degC], defaults to T_op

        Returns
        -------
        dict with cell_voltage, stack_voltage, power_density, stack_power_kw, efficiency
        """
        j = np.asarray(current_density, dtype=float)
        T_K = self._temperature_K(temperature_c)

        V_cell  = self.cell_voltage(j, T_K)
        V_stack = self.N * V_cell
        I       = j * self.A_cell          # A
        P_stack_W = V_stack * I            # W
        P_density  = V_cell * j            # W/cm2

        # Faradaic efficiency (thermodynamic): eta = V_cell * n * F / LHV_H2
        eta = np.where(j > 0.0,
                       (V_cell * self.n * self.F) / self.LHV_H2,
                       0.0)
        eta = np.clip(eta, 0.0, 1.0)

        return {
            "cell_voltage":    V_cell,
            "stack_voltage":   V_stack,
            "power_density":   P_density,        # W/cm2
            "stack_power_kw":  P_stack_W / 1000.0,
            "efficiency":      eta,
        }
