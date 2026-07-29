"""F0a predict interface for EC208 CO2 Geological Sequestration."""
import os, sys, json

sys.path.insert(0, os.path.dirname(__file__))
from model import InjectivityCapacityLookup  # noqa: E402

_HERE = os.path.dirname(__file__)
_PARAMS = os.path.normpath(os.path.join(_HERE, "..", "data", "parameters.json"))


class ComponentModel:
    component_id = "EC208"
    component_name = "CO2 Geological Sequestration"
    fidelity = "F0a -- empirical lookup"
    version = "1.0.0"

    def __init__(self, params_path=_PARAMS):
        with open(params_path) as fh:
            self.params = json.load(fh)
        self.model = InjectivityCapacityLookup(self.params)

    def predict(self, inputs):
        return self.model.predict(inputs)

    def get_info(self):
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {'P_wellhead_bar': 'bar', 'area_km2': 'km2'},
            "outputs": {'injection_rate_kg_s': 'kg/s', 'injection_rate_Mt_yr': 'Mt/yr', 'storage_capacity_Mt': 'Mt'},
            "source": self.params.get("source") or self.params.get("unit", {}).get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.component_id, "-", m.component_name, "[", m.fidelity, "]")
    for k, v in m.predict({'P_wellhead_bar': 150.0, 'area_km2': 100.0}).items():
        print("  {} = {}".format(k, v))
