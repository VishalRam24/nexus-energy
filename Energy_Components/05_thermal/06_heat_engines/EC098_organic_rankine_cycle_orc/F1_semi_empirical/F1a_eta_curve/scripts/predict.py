"""EC098 — Organic Rankine Cycle (ORC) — F1a Efficiency Curve — Standardized Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import ORCF1a


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = ORCF1a(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        inputs:
          T_hot          : heat source temperature (degC)
          T_cold         : heat sink temperature (degC)
          part_load_ratio: part-load ratio (0.3-1.0, default 1.0)
          Q_hot_kw       : optional heat input (kW)
        returns:
          efficiency, power_kw, heat_input_kw, heat_rejection_kw
        """
        T_hot = np.asarray(inputs["T_hot"], dtype=float)
        T_cold = np.asarray(inputs["T_cold"], dtype=float)
        plr = np.asarray(inputs.get("part_load_ratio", 1.0), dtype=float)
        Q_hot_kw = inputs.get("Q_hot_kw", None)

        flows = self._model.power_flows(T_hot, T_cold, plr, Q_hot_kw)
        return {
            "efficiency": flows["efficiency"],
            "power_kw": flows["power_kw"],
            "heat_input_kw": flows["heat_input_kw"],
            "heat_rejection_kw": flows["heat_rejection_kw"],
        }

    def get_info(self) -> dict:
        return {
            "name": "Organic Rankine Cycle (ORC)",
            "ec_id": "EC098",
            "fidelity": "F1a",
            "description": "eta = eta_Carnot * eta_internal * (c0 + c1*PLR); working fluid R245fa",
            "inputs": {
                "T_hot":           {"unit": "degC", "range": [80.0, 300.0]},
                "T_cold":          {"unit": "degC", "range": [10.0, 50.0]},
                "part_load_ratio": {"unit": "-",    "range": [0.3, 1.0], "default": 1.0},
                "Q_hot_kw":        {"unit": "kW",   "range": [0.0, None], "default": "from rated"},
            },
            "outputs": {
                "efficiency":         {"unit": "dimensionless"},
                "power_kw":           {"unit": "kW_e"},
                "heat_input_kw":      {"unit": "kW_th"},
                "heat_rejection_kw":  {"unit": "kW_th"},
            },
            "source": "Quoilin et al. (2013), Ren. Sustain. Energy Rev., 22, 168-186",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    r = model.predict({"T_hot": 150.0, "T_cold": 30.0, "part_load_ratio": 1.0})
    print(f"Rated: eta={float(r['efficiency']):.4f} ({float(r['efficiency'])*100:.2f}%), "
          f"P={float(r['power_kw']):.1f}kW, Q_in={float(r['heat_input_kw']):.1f}kW, "
          f"Q_rej={float(r['heat_rejection_kw']):.1f}kW")
