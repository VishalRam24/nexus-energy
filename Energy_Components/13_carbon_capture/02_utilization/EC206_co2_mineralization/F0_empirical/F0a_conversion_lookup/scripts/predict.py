"""F0a predict interface for EC206 CO2 Mineralization."""
import os, sys, json

sys.path.insert(0, os.path.dirname(__file__))
from model import ConversionLookup  # noqa: E402

_HERE = os.path.dirname(__file__)
_PARAMS = os.path.normpath(os.path.join(_HERE, "..", "data", "parameters.json"))


class ComponentModel:
    component_id = "EC206"
    component_name = "CO2 Mineralization"
    fidelity = "F0a -- empirical lookup"
    version = "1.0.0"

    def __init__(self, params_path=_PARAMS):
        with open(params_path) as fh:
            self.params = json.load(fh)
        self.model = ConversionLookup(self.params)

    def predict(self, inputs):
        return self.model.predict(inputs)

    def get_info(self):
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {'co2_in_kg_s': 'kg/s', 'temperature_C': 'degC'},
            "outputs": {'conversion': '-', 'co2_mineralized_kg_s': 'kg/s', 'carbonate_produced_kg_s': 'kg/s'},
            "source": self.params.get("source") or self.params.get("unit", {}).get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.component_id, "-", m.component_name, "[", m.fidelity, "]")
    for k, v in m.predict({'co2_in_kg_s': 1.0, 'temperature_C': 100.0}).items():
        print("  {} = {}".format(k, v))
