"""F0a predict interface for EC203 Membrane-Based CO2 Separation."""
import os, sys, json

sys.path.insert(0, os.path.dirname(__file__))
from model import SECPressureCurve  # noqa: E402

_HERE = os.path.dirname(__file__)
_PARAMS = os.path.normpath(os.path.join(_HERE, "..", "data", "parameters.json"))


class ComponentModel:
    component_id = "EC203"
    component_name = "Membrane-Based CO2 Separation"
    fidelity = "F0a -- empirical lookup"
    version = "1.0.0"

    def __init__(self, params_path=_PARAMS):
        with open(params_path) as fh:
            self.params = json.load(fh)
        self.model = SECPressureCurve(self.params)

    def predict(self, inputs):
        return self.model.predict(inputs)

    def get_info(self):
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {'co2_feed_kg_s': 'kg/s', 'pressure_ratio': '-', 'recovery': '-'},
            "outputs": {'co2_captured_kg_s': 'kg/s', 'SEC_MJ_kgCO2': 'MJ/kgCO2'},
            "source": self.params.get("source") or self.params.get("unit", {}).get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.component_id, "-", m.component_name, "[", m.fidelity, "]")
    for k, v in m.predict({'co2_feed_kg_s': 10.0, 'pressure_ratio': 10.0}).items():
        print("  {} = {}".format(k, v))
