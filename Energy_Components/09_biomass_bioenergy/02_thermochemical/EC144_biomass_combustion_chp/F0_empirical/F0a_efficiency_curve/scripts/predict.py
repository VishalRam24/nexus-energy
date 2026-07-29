"""Standardized predict interface for EC144 F0a part-load efficiency curve."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import EfficiencyCurve

_DATA = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


class ComponentModel:
    component_id = "EC144"
    component_name = "Biomass Combustion CHP"
    fidelity = "F0a — empirical lookup"
    version = "1.0.0"

    def __init__(self, params_path=_DATA):
        with open(params_path) as f:
            self.params = json.load(f)
        self.curve = EfficiencyCurve(self.params)

    def predict(self, inputs: dict) -> dict:
        PLR = float(inputs.get("PLR", 1.0))
        moisture = float(inputs.get("moisture", 0.0))
        fuel_kw = float(inputs.get("fuel_power_kw", self.curve.Q_rated * PLR))
        eta_el = self.curve.eta_electrical(PLR, moisture)
        eta_th = self.curve.eta_thermal(PLR, moisture)
        return {
            "eta_electrical": eta_el,
            "eta_thermal": eta_th,
            "eta_total": eta_el + eta_th,
            "P_electrical_kw": eta_el * fuel_kw,
            "P_thermal_kw": eta_th * fuel_kw,
            "partload_multiplier": self.curve.partload_multiplier(PLR),
        }

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {"PLR": "0.2-1.0", "moisture": "0-0.6", "fuel_power_kw": ">0"},
            "outputs": ["eta_electrical", "eta_thermal", "eta_total",
                        "P_electrical_kw", "P_thermal_kw"],
            "source": self.params["source"],
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    print(m.predict({"PLR": 1.0, "moisture": 0.15}))
