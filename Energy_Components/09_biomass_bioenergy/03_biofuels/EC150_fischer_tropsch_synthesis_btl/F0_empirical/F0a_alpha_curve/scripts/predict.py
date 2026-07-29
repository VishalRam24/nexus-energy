"""Standardized predict interface for EC150 F0a FT alpha curve."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import AlphaCurve

_DATA = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


class ComponentModel:
    component_id = "EC150"
    component_name = "Fischer-Tropsch Synthesis (BtL)"
    fidelity = "F0a — empirical lookup"
    version = "1.0.0"

    def __init__(self, params_path=_DATA):
        with open(params_path) as f:
            self.params = json.load(f)
        self.curve = AlphaCurve(self.params)

    def predict(self, inputs: dict) -> dict:
        T = float(inputs.get("temperature_degC", self.curve.T_ref))
        syngas_mol = float(inputs.get("syngas_CO_mol", 1.0))
        a = self.curve.alpha(T)
        co = self.curve.co_conversion(T)
        diesel_sel = self.curve.diesel_selectivity(T)
        return {
            "alpha": a,
            "CO_conversion": co,
            "diesel_selectivity": diesel_sel,
            "diesel_yield_proxy": co * diesel_sel,
            "CO_converted_mol": co * syngas_mol,
        }

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {"temperature_degC": "180-350", "syngas_CO_mol": ">0"},
            "outputs": ["alpha", "CO_conversion", "diesel_selectivity",
                        "diesel_yield_proxy", "CO_converted_mol"],
            "source": self.params["source"],
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    print(m.predict({"temperature_degC": 230, "syngas_CO_mol": 1000}))
