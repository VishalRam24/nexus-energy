"""F0a predict interface for EC090 Solar Water Heater Combi System.

Standardized ComponentModel wrapper around the empirical solar fraction table model.
NumPy only. See model.py for the empirical relation and data source.
"""
import json, os, sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from model import SolarFractionModel  # noqa: E402


class ComponentModel:
    component_id = "EC090"
    component_name = "Solar Water Heater Combi System"
    fidelity = "F0a — empirical lookup"
    version = "1.0.0"

    def __init__(self, params_path=None):
        if params_path is None:
            params_path = os.path.join(_HERE, "..", "data", "parameters.json")
        with open(params_path) as fh:
            self.params = json.load(fh)
        self.model = SolarFractionModel(self.params)

    def predict(self, inputs):
        return self.model.predict(inputs)

    def get_info(self):
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {'irradiance_W_m2': 'W/m2', 'Q_demand_W': 'W'},
            "outputs": {'f_solar': '-', 'Q_solar_W': 'W', 'Q_aux_input_W': 'W'},
            "source": self.params.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    print("sample:", m.predict({"irradiance_W_m2": 600.0}))
