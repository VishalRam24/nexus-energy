"""EC040 -- Hydrogen-Bromine Flow Battery -- F1a SOC-only (Nernst OCV)"""
import math


class H2BrFlowBatteryModel:
    """
    Hydrogen-Bromine (H2-Br) flow battery SOC-only model.

    Positive: Br2 + 2e- <-> 2Br-   E0_pos = +1.09 V vs SHE
    Negative: 2H+ + 2e- <-> H2      E0_neg = 0.0 V
    Cell E0 = 1.09 V, n=2

    Nernst OCV for single cell:
        E_cell = E0 + (R*T)/(n*F) * ln(SOC / (1 - SOC))

    Terminal voltage (I > 0 = discharge):
        V = V_stack - I * R_int_stack
    """

    F_CONST = 96485.0
    R_CONST = 8.314

    def __init__(self, params: dict):
        self.E0 = float(params.get("E0", 1.09))
        self.n = int(params.get("n", 2))
        self.T = float(params.get("T", 298.15))
        self.R_int_area = float(params.get("R_int_area", 0.5))
        self.N_cells = int(params.get("N_cells", 30))
        self.A_cell = float(params.get("A_cell", 800.0))
        self.Q_nom_Ah = float(params.get("Q_nom_Ah", 80.0))
        self._R_int_stack = self.N_cells * self.R_int_area / self.A_cell

    def _nernst_cell(self, soc: float) -> float:
        soc = max(0.001, min(0.999, soc))
        return self.E0 + (self.R_CONST * self.T / (self.n * self.F_CONST)) * math.log(soc / (1.0 - soc))

    def evaluate(self, soc: float, I: float = 0.0, dt: float = 0.0) -> dict:
        soc = max(0.001, min(0.999, float(soc)))
        I = float(I)
        dt = float(dt)

        E_cell = self._nernst_cell(soc)
        V_stack_ocv = E_cell * self.N_cells
        V_stack_term = V_stack_ocv - I * self._R_int_stack
        P_stack = V_stack_term * I

        soc_new = soc
        if dt > 0.0:
            soc_new = max(0.001, min(0.999, soc - I * dt / (self.Q_nom_Ah * 3600.0)))

        eta = V_stack_term / V_stack_ocv if V_stack_ocv > 0 and I > 0 else 1.0

        return {
            "V_cell_ocv": round(E_cell, 6),
            "V_stack_ocv": round(V_stack_ocv, 4),
            "V_stack_terminal": round(V_stack_term, 4),
            "P_stack": round(P_stack, 2),
            "SOC_new": round(soc_new, 6),
            "efficiency": round(eta, 6),
        }
