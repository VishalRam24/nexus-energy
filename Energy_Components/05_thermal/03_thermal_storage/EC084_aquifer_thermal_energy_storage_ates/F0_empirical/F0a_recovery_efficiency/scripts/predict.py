"""F0a predict interface for EC084 Aquifer Thermal Energy Storage (ATES).

Standardized ComponentModel wrapper around the empirical storage efficiency lookup model.
NumPy only. See model.py for the empirical relation and data source.
"""
import json, os, sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from model import StoreEffModel  # noqa: E402


class ComponentModel:
    component_id = "EC084"
    component_name = "Aquifer Thermal Energy Storage (ATES)"
    fidelity = "F0a — empirical lookup"
    version = "1.0.0"

    def __init__(self, params_path=None):
        if params_path is None:
            params_path = os.path.join(_HERE, "..", "data", "parameters.json")
        with open(params_path) as fh:
            self.params = json.load(fh)
        self.model = StoreEffModel(self.params)

    def predict(self, inputs):
        return self.model.predict(inputs)

    def get_info(self):
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {'T_warm_C': 'driving variable', 'E_charge_kWh': 'kWh'},
            "outputs": {'efficiency': '-', 'E_discharge_kWh': 'kWh'},
            "source": self.params.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    print("sample:", m.predict({"T_warm_C": 25.0, "E_charge_kWh": 1000.0}))
