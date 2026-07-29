"""EC006 -- DMFC -- F1a Polarization Curve Model"""
import math


class DMFCPolarizationModel:
    """
    Direct Methanol Fuel Cell polarization curve.

    CH3OH + H2O -> CO2 + 6H+ + 6e-   (n=6)

    OCV is reduced from E_rev=1.21 V to ~0.6 V due to methanol crossover
    through the Nafion membrane (mixed-potential effect).

    V = OCV - V_act - V_ohm - V_conc

    V_act  = (RT / (alpha * n * F)) * ln(j / i0)
    V_ohm  = j * R_ohm
    V_conc = (RT / (n * F)) * ln(j_L / (j_L - j))
    """

    F_CONST = 96485.0
    R_CONST = 8.314
    N_ELEC  = 6  # 6 electrons per methanol molecule

    def __init__(self, params: dict):
        self.T       = float(params.get("T", 343.15))
        self.E_rev   = float(params.get("E_rev", 1.21))
        self.OCV     = float(params.get("OCV", 0.6))
        self.i0      = float(params.get("i0", 1e-5))
        self.R_ohm   = float(params.get("R_ohm", 0.4))
        self.j_L     = float(params.get("j_L", 0.4))
        self.alpha   = float(params.get("alpha", 0.5))
        self.n_cells = int(params.get("n_cells", 1))
        self.area    = float(params.get("area", 100.0))

    def _cell_voltage(self, j: float) -> float:
        if j <= 0.0:
            return self.OCV
        if j >= self.j_L:
            return 0.0

        RT_nF  = self.R_CONST * self.T / (self.N_ELEC * self.F_CONST)
        V_act  = (RT_nF / self.alpha) * math.log(j / self.i0)
        V_ohm  = j * self.R_ohm
        V_conc = RT_nF * math.log(self.j_L / (self.j_L - j))
        return max(self.OCV - V_act - V_ohm - V_conc, 0.0)

    def evaluate(self, j: float, **kwargs) -> dict:
        """
        Inputs
        ------
        j : float -- current density (A/cm^2)

        Outputs
        -------
        V_cell, V_stack, P_density, P_stack, efficiency (vs E_rev), efficiency_ocv (vs OCV)
        """
        j       = float(j)
        V_cell  = self._cell_voltage(j)
        V_stack = V_cell * self.n_cells
        P_dens  = V_cell * j
        P_stack = P_dens * self.area * self.n_cells
        eta_rev = V_cell / self.E_rev if self.E_rev > 0 else 0.0
        eta_ocv = V_cell / self.OCV if self.OCV > 0 else 0.0

        return {
            "V_cell":         round(V_cell, 6),
            "V_stack":        round(V_stack, 6),
            "P_density":      round(P_dens, 6),
            "P_stack":        round(P_stack, 4),
            "efficiency":     round(eta_rev, 6),
            "efficiency_ocv": round(eta_ocv, 6),
        }
