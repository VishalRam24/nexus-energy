"""EC101 -- CCGT -- F1b Part-Load + Ambient -- Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import CCGTF1b


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = CCGTF1b(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        inputs:
            PLR       : float or array [0.4 - 1.0]
            T_ambient : float or array [K] (default 288.15)
            P_ambient : float or array [kPa] (default 101.325)
        returns:
            efficiency_combined : combined cycle efficiency [-]
            efficiency_gt       : GT efficiency [-]
            efficiency_st       : ST effective efficiency [-]
            power_output_kw     : combined electrical output [kW]
            heat_rate_kj_kwh    : combined heat rate [kJ/kWh]
            exhaust_temp_K      : stack temperature [K]
        """
        PLR   = np.asarray(inputs["PLR"], dtype=float)
        T_amb = np.asarray(inputs.get("T_ambient", 288.15), dtype=float)
        P_amb = np.asarray(inputs.get("P_ambient", 101.325), dtype=float)

        return {
            "efficiency_combined": self._model.efficiency_combined(PLR, T_amb, P_amb),
            "efficiency_gt":       self._model.efficiency_gt_out(PLR, T_amb, P_amb),
            "efficiency_st":       self._model.efficiency_st_out(PLR, T_amb, P_amb),
            "power_output_kw":     self._model.power_output_kw(PLR, T_amb, P_amb),
            "heat_rate_kj_kwh":    self._model.heat_rate_kj_kwh(PLR, T_amb, P_amb),
            "exhaust_temp_K":      self._model.exhaust_temp_k(PLR),
        }

    def get_info(self) -> dict:
        return {
            "name": "Combined Cycle Gas Turbine (CCGT)",
            "ec_id": "EC101",
            "fidelity": "F1b",
            "model": "Part-Load + Ambient (GT + ST separated)",
            "description": (
                "GT part-load + ambient correction; ST bottoming cycle from exhaust heat; "
                "eta_cc = eta_GT + (1-eta_GT)*eta_ST_eff"
            ),
            "inputs": {
                "PLR":       {"unit": "-", "range": [0.4, 1.0]},
                "T_ambient": {"unit": "K", "range": [243.15, 323.15], "default": 288.15},
                "P_ambient": {"unit": "kPa", "range": [80.0, 110.0], "default": 101.325},
            },
            "outputs": {
                "efficiency_combined": {"unit": "-"},
                "efficiency_gt":       {"unit": "-"},
                "efficiency_st":       {"unit": "-"},
                "power_output_kw":     {"unit": "kW"},
                "heat_rate_kj_kwh":    {"unit": "kJ/kWh"},
                "exhaust_temp_K":      {"unit": "K"},
            },
            "source": "Kehlhofer et al. (2009); Chase (2001)",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    print("EC101 F1b -- ISO conditions (PLR=1.0, 288.15K, 101.325kPa):")
    r = model.predict({"PLR": 1.0})
    for k, v in r.items():
        print(f"  {k}: {float(v):.4f}")
    print("\n50% load, hot day:")
    r = model.predict({"PLR": 0.5, "T_ambient": 313.15})
    for k, v in r.items():
        print(f"  {k}: {float(v):.4f}")
