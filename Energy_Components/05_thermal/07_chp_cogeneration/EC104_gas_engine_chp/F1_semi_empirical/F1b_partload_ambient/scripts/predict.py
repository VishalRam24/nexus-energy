"""EC104 -- Gas Engine CHP -- F1b Part-Load + Ambient -- Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import GasEngineCHPF1b


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = GasEngineCHPF1b(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        inputs:
            PLR        : float or array [0.5 - 1.0]
            T_ambient  : float or array [degC] (default 25.0)
        returns:
            efficiency_electrical : electrical efficiency [-]
            efficiency_thermal    : thermal efficiency [-]
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
            "name": "Gas Engine CHP",
            "ec_id": "EC104",
            "fidelity": "F1b",
            "model": "Part-Load + Ambient Temperature",
            "description": (
                "eta_el(PLR) = eta_el_rated * (a + b*PLR + c*PLR^2); "
                "eta_th(PLR) = eta_th_rated * (th_a + th_b*PLR); "
                "With temperature derating above 25 degC"
            ),
            "inputs": {
                "PLR":       {"unit": "-", "range": [0.5, 1.0]},
                "T_ambient": {"unit": "degC", "range": [-20, 50], "default": 25.0},
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
            "source": "US EPA CHP Catalog (2017); ASUE BHKW-Kenndaten (2011)",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    print("EC104 F1b -- Standard (PLR=1.0, 25C):")
    r = model.predict({"PLR": 1.0})
    for k, v in r.items():
        print(f"  {k}: {float(v):.4f}")
    print("\nPart-load, hot (PLR=0.6, 45C):")
    r = model.predict({"PLR": 0.6, "T_ambient": 45.0})
    for k, v in r.items():
        print(f"  {k}: {float(v):.4f}")
