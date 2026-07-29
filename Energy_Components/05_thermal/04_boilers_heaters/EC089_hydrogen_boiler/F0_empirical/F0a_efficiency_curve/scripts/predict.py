"""F0a predict interface for EC089 Hydrogen Boiler (100% H2).

Standardized ComponentModel wrapper around the empirical part-load efficiency curve model.
NumPy only. See model.py for the empirical relation and data source.
"""
import json, os, sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from model import EffCurveModel  # noqa: E402


class ComponentModel:
    component_id = "EC089"
    component_name = "Hydrogen Boiler (100% H2)"
    fidelity = "F0a — empirical lookup"
    version = "1.0.0"

    def __init__(self, params_path=None):
        if params_path is None:
            params_path = os.path.join(_HERE, "..", "data", "parameters.json")
        with open(params_path) as fh:
            self.params = json.load(fh)
        self.model = EffCurveModel(self.params)

    def predict(self, inputs):
        return self.model.predict(inputs)

    def get_info(self):
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {'part_load_ratio': '0-1 (-)'},
            "outputs": {'efficiency': '-', 'Q_out_kW': 'kW', 'Q_in_kW': 'kW'},
            "source": self.params.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    print("sample:", m.predict({"part_load_ratio": 0.5}))
