"""F0a predict interface for EC200 Oxy-Fuel Combustion Capture."""
import os, sys, json

sys.path.insert(0, os.path.dirname(__file__))
from model import EnergyPenaltyLookup  # noqa: E402

_HERE = os.path.dirname(__file__)
_PARAMS = os.path.normpath(os.path.join(_HERE, "..", "data", "parameters.json"))


class ComponentModel:
    component_id = "EC200"
    component_name = "Oxy-Fuel Combustion Capture"
    fidelity = "F0a -- empirical lookup"
    version = "1.0.0"

    def __init__(self, params_path=_PARAMS):
        with open(params_path) as fh:
            self.params = json.load(fh)
        self.model = EnergyPenaltyLookup(self.params)

    def predict(self, inputs):
        return self.model.predict(inputs)

    def get_info(self):
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {'fuel_rate': 'kg/s', 'load': '-'},
            "outputs": {'o2_demand_kg_s': 'kg/s', 'co2_captured_kg_s': 'kg/s', 'specific_penalty_kWh_tCO2': 'kWh/tCO2'},
            "source": self.params.get("source") or self.params.get("unit", {}).get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.component_id, "-", m.component_name, "[", m.fidelity, "]")
    for k, v in m.predict({'fuel_rate': 10.0, 'load': 1.0}).items():
        print("  {} = {}".format(k, v))
