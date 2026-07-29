"""EC098 -- ORC -- F1b Part-Load + Condenser Ambient -- Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import ORCF1b


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = ORCF1b(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        inputs:
            T_heat_source  : float or array [degC, 80-300]
            T_condenser    : float or array [degC, 15-55]
            PLR            : float or array [0.3 - 1.0]
            heat_input_kw  : float or array [kW] (optional)
        returns:
            efficiency          : net ORC thermal efficiency [-]
            power_output_kw     : electrical output [kW]
            heat_rejection_kw   : heat rejected to condenser [kW]
            specific_work_kj_kg : approximate specific work [kJ/kg]
        """
        T_hot  = np.asarray(inputs.get("T_heat_source", 150.0), dtype=float)
        T_cond = np.asarray(inputs.get("T_condenser", 30.0), dtype=float)
        PLR    = np.asarray(inputs.get("PLR", 1.0), dtype=float)
        Q_hot  = inputs.get("heat_input_kw", None)

        return {
            "efficiency":          self._model.efficiency(T_hot, T_cond, PLR),
            "power_output_kw":     self._model.power_output_kw(T_hot, T_cond, PLR, Q_hot),
            "heat_rejection_kw":   self._model.heat_rejection_kw(T_hot, T_cond, PLR, Q_hot),
            "specific_work_kj_kg": self._model.specific_work_kj_kg(T_hot, T_cond, PLR),
        }

    def get_info(self) -> dict:
        return {
            "name": "Organic Rankine Cycle (ORC)",
            "ec_id": "EC098",
            "fidelity": "F1b",
            "model": "Part-Load + Condenser Ambient Effect",
            "description": (
                "eta = eta_carnot * eta_internal * f_PLR(PLR) * f_T(T_cond); "
                "ORC efficiency very sensitive to condenser temperature"
            ),
            "inputs": {
                "T_heat_source":  {"unit": "degC", "range": [80, 300], "default": 150.0},
                "T_condenser":    {"unit": "degC", "range": [15, 55], "default": 30.0},
                "PLR":            {"unit": "-", "range": [0.3, 1.0], "default": 1.0},
                "heat_input_kw":  {"unit": "kW", "range": [50, 2000], "default": "auto"},
            },
            "outputs": {
                "efficiency":          {"unit": "-"},
                "power_output_kw":     {"unit": "kW"},
                "heat_rejection_kw":   {"unit": "kW"},
                "specific_work_kj_kg": {"unit": "kJ/kg"},
            },
            "source": "Quoilin et al. (2013); Manente et al. (2013)",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    print("EC098 F1b -- Design conditions (T_hot=150C, T_cond=30C, PLR=1.0):")
    r = model.predict({"T_heat_source": 150.0, "T_condenser": 30.0, "PLR": 1.0})
    for k, v in r.items():
        print(f"  {k}: {float(v):.4f}")
    print("\nOff-design (T_hot=120C, T_cond=45C, PLR=0.5):")
    r = model.predict({"T_heat_source": 120.0, "T_condenser": 45.0, "PLR": 0.5})
    for k, v in r.items():
        print(f"  {k}: {float(v):.4f}")
