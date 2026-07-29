"""EC007 -- RFC -- F1a Polarization Curve Model (bidirectional)"""
import math


class RFCPolarizationModel:
    """
    Reversible Fuel Cell: operates as FC (j > 0) or electrolyzer (j < 0).

    Fuel Cell mode (j > 0):
        V = E_rev - V_act - V_ohm - V_conc
        Voltage decreases with increasing j.

    Electrolyzer mode (j < 0):
        V = E_rev + V_act + V_ohm + V_conc
        Voltage increases (above E_rev) with increasing |j|.

    Convention: j positive = discharge (FC), j negative = charge (EL).
    All overpotentials use |j|.
    """

    F_CONST = 96485.0
    R_CONST = 8.314
    N_ELEC  = 2

    def __init__(self, params: dict):
        self.T       = float(params.get("T", 353.15))
        self.E_rev   = float(params.get("E_rev", 1.23))
        self.i0      = float(params.get("i0", 1e-3))
        self.R_ohm   = float(params.get("R_ohm", 0.3))
        self.j_L_fc  = float(params.get("j_L_fc", 1.0))
        self.j_L_el  = float(params.get("j_L_el", 2.0))
        self.alpha   = float(params.get("alpha", 0.5))
        self.n_cells = int(params.get("n_cells", 1))
        self.area    = float(params.get("area", 100.0))

    def _overpotentials(self, abs_j: float, j_L: float) -> float:
        """Sum of activation + ohmic + concentration losses for |j|."""
        RT_nF  = self.R_CONST * self.T / (self.N_ELEC * self.F_CONST)
        V_act  = (RT_nF / self.alpha) * math.log(abs_j / self.i0)
        V_ohm  = abs_j * self.R_ohm
        V_conc = RT_nF * math.log(j_L / (j_L - abs_j))
        return V_act + V_ohm + V_conc

    def evaluate(self, j: float, **kwargs) -> dict:
        """
        Inputs
        ------
        j : float -- current density (A/cm^2).
                     Positive = FC mode, negative = EL mode, 0 = OCV.

        Outputs
        -------
        V_cell    : cell voltage (V)
        V_stack   : stack voltage (V)
        mode      : 'FC', 'EL', or 'OCV'
        P_density : power density (W/cm^2), positive = power output (FC), negative = input (EL)
        P_stack   : stack power (W)
        efficiency: |V_cell / E_rev|
        """
        j = float(j)

        if j == 0.0:
            V_cell = self.E_rev
            mode   = "OCV"
        elif j > 0.0:
            # Fuel cell mode
            if j >= self.j_L_fc:
                V_cell = 0.0
            else:
                losses = self._overpotentials(j, self.j_L_fc)
                V_cell = max(self.E_rev - losses, 0.0)
            mode = "FC"
        else:
            # Electrolyzer mode: j is negative
            abs_j = abs(j)
            if abs_j >= self.j_L_el:
                V_cell = self.E_rev + self._overpotentials(self.j_L_el * 0.9999, self.j_L_el)
            else:
                losses = self._overpotentials(abs_j, self.j_L_el)
                V_cell = self.E_rev + losses
            mode = "EL"

        V_stack   = V_cell * self.n_cells
        P_density = V_cell * j          # negative in EL mode (power consumed)
        P_stack   = P_density * self.area * self.n_cells
        eta       = abs(V_cell / self.E_rev) if self.E_rev > 0 else 0.0

        return {
            "V_cell":     round(V_cell, 6),
            "V_stack":    round(V_stack, 6),
            "mode":       mode,
            "P_density":  round(P_density, 6),
            "P_stack":    round(P_stack, 4),
            "efficiency": round(eta, 6),
        }
