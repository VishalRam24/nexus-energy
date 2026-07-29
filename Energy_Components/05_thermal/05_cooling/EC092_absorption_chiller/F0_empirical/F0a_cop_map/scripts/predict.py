"""F0a predict interface for EC092 Single-Effect LiBr-H2O Absorption Chiller.

Standardized ComponentModel wrapper around the empirical COP map model.
NumPy only. See model.py for the empirical relation and data source.
"""
import json, os, sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from model import CopMapModel  # noqa: E402


class ComponentModel:
    component_id = "EC092"
    component_name = "Single-Effect LiBr-H2O Absorption Chiller"
    fidelity = "F0a — empirical lookup"
    version = "1.0.0"

    def __init__(self, params_path=None):
        if params_path is None:
            params_path = os.path.join(_HERE, "..", "data", "parameters.json")
        with open(params_path) as fh:
            self.params = json.load(fh)
        self.model = CopMapModel(self.params)

    def predict(self, inputs):
        return self.model.predict(inputs)

    def get_info(self):
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {'T_gen_C': 'driving variable', 'part_load_ratio': '0-1 (-)'},
            "outputs": {'COP': '-', 'Q_cool_kW': 'kW_th', 'W_in_kW': 'kW'},
            "source": self.params.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    print("sample:", m.predict({"T_gen_C": 90.0}))
