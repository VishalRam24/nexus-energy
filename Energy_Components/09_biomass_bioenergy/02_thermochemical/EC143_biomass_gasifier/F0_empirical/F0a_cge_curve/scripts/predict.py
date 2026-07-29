"""Standardized predict interface for EC143 F0a cold-gas-efficiency curve."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import CGECurve

_DATA = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


class ComponentModel:
    component_id = "EC143"
    component_name = "Biomass Gasifier"
    fidelity = "F0a — empirical lookup"
    version = "1.0.0"

    def __init__(self, params_path=_DATA):
        with open(params_path) as f:
            self.params = json.load(f)
        self.curve = CGECurve(self.params)

    def predict(self, inputs: dict) -> dict:
        feedstock = inputs.get("feedstock", "wood")
        ER = float(inputs.get("equivalence_ratio", self.curve.ER_design))
        moisture = float(inputs.get("moisture", 0.0))
        feed_kw = float(inputs.get("fuel_power_kw", 1.0))
        cge = self.curve.cge(feedstock, ER, moisture)
        return {
            "cold_gas_efficiency": cge,
            "syngas_power_kw": cge * feed_kw,
            "cge_at_design_ER": self.curve.cge_vs_er(self.curve.ER_design),
            "fuel_factor": self.curve.fuel_factor(feedstock),
            "LHV_syngas_MJ_Nm3": self.curve.lhv_syngas,
        }

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {"feedstock": list(self.curve.feedstocks),
                       "equivalence_ratio": "0.15-0.45", "moisture": "0-0.5",
                       "fuel_power_kw": ">0"},
            "outputs": ["cold_gas_efficiency", "syngas_power_kw", "fuel_factor"],
            "source": self.params["source"],
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    print(m.predict({"feedstock": "pine", "equivalence_ratio": 0.25, "fuel_power_kw": 1000}))
