"""EC004 -- PAFC -- F1a Polarization Curve Model"""
import math


class PAFCPolarizationModel:
    """
    Phosphoric Acid Fuel Cell polarization curve.

    V = E_nernst - V_act - V_ohm - V_conc

    V_act  = (RT / (alpha * n * F)) * ln(j / i0)
    V_ohm  = j * R_ohm
    V_conc = (RT / (n * F)) * ln(j_L / (j_L - j))

    Operating temperature ~423 K (150 degC).
    """

    F_CONST = 96485.0  # C/mol
    R_CONST = 8.314    # J/(mol*K)
    N_ELEC  = 2        # electrons per H2 molecule

    def __init__(self, params: dict):
        self.T      = float(params.get("T", 423.15))
        self.E_rev  = float(params.get("E_rev", 1.1))
        self.i0     = float(params.get("i0", 1e-4))
        self.R_ohm  = float(params.get("R_ohm", 0.5))
        self.j_L    = float(params.get("j_L", 0.8))
        self.alpha  = float(params.get("alpha", 0.5))
        self.n_cells = int(params.get("n_cells", 1))
        self.area   = float(params.get("area", 100.0))

    def _cell_voltage(self, j: float) -> float:
        """Compute single-cell voltage for current density j (A/cm^2)."""
        if j <= 0.0:
            return self.E_rev
        if j >= self.j_L:
            return 0.0

        RT_nF = self.R_CONST * self.T / (self.N_ELEC * self.F_CONST)

        V_act  = (RT_nF / self.alpha) * math.log(j / self.i0)
        V_ohm  = j * self.R_ohm
        V_conc = RT_nF * math.log(self.j_L / (self.j_L - j))

        V = self.E_rev - V_act - V_ohm - V_conc
        return max(V, 0.0)

    def evaluate(self, j: float, **kwargs) -> dict:
        """
        Inputs
        ------
        j : float -- current density (A/cm^2)

        Outputs
        -------
        V_cell   : float -- single-cell voltage (V)
        V_stack  : float -- stack voltage (V)
        P_density: float -- power density (W/cm^2)
        P_stack  : float -- stack power (W)
        efficiency: float -- voltage efficiency (-)
        """
        j = float(j)
        V_cell    = self._cell_voltage(j)
        V_stack   = V_cell * self.n_cells
        P_density = V_cell * j
        P_stack   = P_density * self.area * self.n_cells
        eta       = V_cell / self.E_rev if self.E_rev > 0 else 0.0

        return {
            "V_cell":     round(V_cell, 6),
            "V_stack":    round(V_stack, 6),
            "P_density":  round(P_density, 6),
            "P_stack":    round(P_stack, 4),
            "efficiency": round(eta, 6),
        }
