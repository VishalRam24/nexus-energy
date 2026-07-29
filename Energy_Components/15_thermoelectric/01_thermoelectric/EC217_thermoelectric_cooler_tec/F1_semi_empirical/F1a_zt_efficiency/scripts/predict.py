"""EC217 — Thermoelectric Cooler (TEC) — F1a — Standardized Predict Interface"""
import json, os, sys
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from model import TECF1a


class ComponentModel:
    component_id = "EC217"

    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = TECF1a(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        Inputs:
            Tc_K : float or array [233, 320] — cold side temperature [K]
            Th_K : float or array [280, 360] — hot side temperature [K]
            I_A  : float or array [0.1, 5.0] — operating current [A], default optimal
        Returns:
            COP_carnot, COP_ZT, COP_physical, Q_cool_W, W_input_W,
            eta_zt, ZT_eff, I_optimal_A
        """
        Tc = np.asarray(inputs.get("Tc_K", 280.0), dtype=float)
        Th = np.asarray(inputs.get("Th_K", 310.0), dtype=float)

        I_opt = self._model.I_optimal(Tc, Th)
        I = np.asarray(inputs.get("I_A", I_opt), dtype=float)

        return {
            "COP_carnot": self._model.COP_carnot(Tc, Th),
            "COP_ZT": self._model.COP(Tc, Th),
            "COP_physical": self._model.COP_physical(Tc, Th, I),
            "Q_cool_W": self._model.Q_cool_physical(Tc, Th, I),
            "W_input_W": self._model.W_input(Tc, Th, I),
            "eta_zt": self._model.eta_zt(Tc, Th),
            "ZT_eff": self._model.ZT_at_T(Tc, Th),
            "I_optimal_A": I_opt,
        }

    def get_info(self) -> dict:
        u = self.params["unit"]
        return {
            "name": "Thermoelectric Cooler (TEC) — Peltier",
            "ec_id": self.component_id,
            "fidelity": "F1a",
            "description": (
                "Bi2Te3 Peltier cooler. ZT=0.7, COP = Tc/(Th-Tc)*eta_zt. "
                "Q_cool = N*[alpha*Tc*I - 0.5*I^2*R/N - K/N*dT]. Physical + ZT models."
            ),
            "inputs": {
                "Tc_K": {"unit": "K", "range": [233, 320]},
                "Th_K": {"unit": "K", "range": [280, 360]},
                "I_A": {"unit": "A", "range": [0.1, 5.0], "default": "optimal"},
            },
            "outputs": {
                "COP_carnot": {"unit": "dimensionless"},
                "COP_ZT": {"unit": "dimensionless"},
                "COP_physical": {"unit": "dimensionless"},
                "Q_cool_W": {"unit": "W"},
                "W_input_W": {"unit": "W"},
                "eta_zt": {"unit": "dimensionless"},
                "ZT_eff": {"unit": "dimensionless"},
                "I_optimal_A": {"unit": "A"},
            },
            "params": {
                "ZT": str(u["ZT"]["value"]),
                "N_couples": str(u["N_couples"]["value"]),
                "alpha": f"{u['alpha_Seebeck']['value']} V/K",
                "R_module": f"{u['R_module']['value']} ohm",
                "K_module": f"{u['K_module']['value']} W/K",
                "material": "Bi2Te3",
            },
            "source": "Rowe (2006); Goldsmid (2010)",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    for dT in [5, 10, 20, 30, 40]:
        Tc = 280.0
        Th = Tc + dT
        r = model.predict({"Tc_K": Tc, "Th_K": Th})
        print(f"Tc={Tc:.0f}K Th={Th:.0f}K dT={dT}: "
              f"COP_ZT={float(r['COP_ZT']):.3f}  COP_phys={float(r['COP_physical']):.3f}  "
              f"Q_cool={float(r['Q_cool_W']):.2f} W  W_in={float(r['W_input_W']):.2f} W")
