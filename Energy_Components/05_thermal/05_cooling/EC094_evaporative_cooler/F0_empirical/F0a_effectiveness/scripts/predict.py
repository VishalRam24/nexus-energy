"""F0a predict interface for EC094 Evaporative Cooler (Direct).

Standardized ComponentModel wrapper around the empirical effectiveness model model.
NumPy only. See model.py for the empirical relation and data source.
"""
import json, os, sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from model import EvapModel  # noqa: E402


class ComponentModel:
    component_id = "EC094"
    component_name = "Evaporative Cooler (Direct)"
    fidelity = "F0a — empirical lookup"
    version = "1.0.0"

    def __init__(self, params_path=None):
        if params_path is None:
            params_path = os.path.join(_HERE, "..", "data", "parameters.json")
        with open(params_path) as fh:
            self.params = json.load(fh)
        self.model = EvapModel(self.params)

    def predict(self, inputs):
        return self.model.predict(inputs)

    def get_info(self):
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {'T_db_C': 'degC', 'T_wb_C': 'degC', 'm_air_kg_s': 'kg/s'},
            "outputs": {'T_out_C': 'degC', 'Q_cool_W': 'W', 'COP': '-'},
            "source": self.params.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    print("sample:", m.predict({"T_db_C": 35.0, "T_wb_C": 20.0}))
