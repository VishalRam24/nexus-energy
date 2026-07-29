"""EC106 -- SOFC-Based Fuel Cell CHP -- F1b Part-Load + Ambient -- Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import FuelCellCHPF1b


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = FuelCellCHPF1b(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        inputs:
            PLR        : float or array [0.3 - 1.0]
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
            "name": "SOFC-Based Fuel Cell CHP",
            "ec_id": "EC106",
            "fidelity": "F1b",
            "model": "Part-Load + Ambient Temperature",
            "description": (
                "eta_el(PLR) = eta_el_rated * (a + b*PLR + c*PLR^2); "
                "SOFC characteristic: high part-load electrical efficiency; "
                "With ambient temperature derating above 25 degC"
            ),
            "inputs": {
                "PLR":       {"unit": "-", "range": [0.3, 1.0]},
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
            "source": "US DOE NETL SOFC Program Review (2020); US EPA CHP Catalog (2017)",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    print("EC106 F1b -- Rated (PLR=1.0, 25C):")
    r = model.predict({"PLR": 1.0})
    for k, v in r.items():
        print(f"  {k}: {float(v):.4f}")
    print("\nPart-load (PLR=0.4, 35C):")
    r = model.predict({"PLR": 0.4, "T_ambient": 35.0})
    for k, v in r.items():
        print(f"  {k}: {float(v):.4f}")
