"""EC107 -- Micro-CHP Stirling Engine -- F1b Part-Load + Ambient -- Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import MicroCHPStirlingF1b


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = MicroCHPStirlingF1b(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        inputs:
            PLR        : float or array [0.4 - 1.0]
            T_ambient  : float or array [degC] (default 25.0)
        returns:
            efficiency_electrical : electrical efficiency [-]
            efficiency_thermal    : thermal recovery efficiency [-]
            efficiency_total      : total first-law efficiency [-]
            power_electrical_kw   : electrical output [kW_e]
            heat_recovery_kw      : thermal recovery [kW_th]
            fuel_input_kw         : fuel input [kW_fuel]
            heat_to_power_ratio   : HPR = Q_th / P_el [-]
        """
        PLR   = np.asarray(inputs["PLR"], dtype=float)
        T_amb = np.asarray(inputs.get("T_ambient", 25.0), dtype=float)

        return {
            "efficiency_electrical": self._model.eta_electrical(PLR, T_amb),
            "efficiency_thermal":    self._model.eta_thermal(PLR),
            "efficiency_total":      self._model.eta_total(PLR, T_amb),
            "power_electrical_kw":   self._model.power_electrical_kw(PLR, T_amb),
            "heat_recovery_kw":      self._model.heat_recovery_kw(PLR, T_amb),
            "fuel_input_kw":         self._model.fuel_input_kw(PLR, T_amb),
            "heat_to_power_ratio":   self._model.heat_to_power_ratio(PLR, T_amb),
        }

    def get_info(self) -> dict:
        return {
            "name": "Micro-CHP Stirling Engine",
            "ec_id": "EC107",
            "fidelity": "F1b",
            "model": "Part-Load + Ambient Temperature",
            "description": (
                "eta_el(PLR) = eta_el_rated * (a + b*PLR + c*PLR^2); "
                "Stirling: high HPR (~6-7), low eta_el (~12%), high eta_th (~80%); "
                "T_cold=ambient drives efficiency derating above 25 degC"
            ),
            "inputs": {
                "PLR":       {"unit": "-", "range": [0.4, 1.0]},
                "T_ambient": {"unit": "degC", "range": [-20, 45], "default": 25.0},
            },
            "outputs": {
                "efficiency_electrical": {"unit": "-"},
                "efficiency_thermal":    {"unit": "-"},
                "efficiency_total":      {"unit": "-"},
                "power_electrical_kw":   {"unit": "kW_e"},
                "heat_recovery_kw":      {"unit": "kW_th"},
                "fuel_input_kw":         {"unit": "kW_fuel"},
                "heat_to_power_ratio":   {"unit": "-"},
            },
            "source": "Hawkes & Leach (2007) Energy; Lund et al. (2016) Elsevier",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    print("EC107 F1b -- Rated (PLR=1.0, 25C):")
    r = model.predict({"PLR": 1.0})
    for k, v in r.items():
        print(f"  {k}: {float(v):.4f}")
    print("\nPart-load (PLR=0.5, 35C):")
    r = model.predict({"PLR": 0.5, "T_ambient": 35.0})
    for k, v in r.items():
        print(f"  {k}: {float(v):.4f}")
