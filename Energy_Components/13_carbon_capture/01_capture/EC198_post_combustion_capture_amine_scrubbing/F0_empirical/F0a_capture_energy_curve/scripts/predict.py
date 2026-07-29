"""F0a predict interface for EC198 Post-Combustion Capture (Amine Scrubbing)."""
import os, sys, json

sys.path.insert(0, os.path.dirname(__file__))
from model import CaptureEnergyCurve  # noqa: E402

_HERE = os.path.dirname(__file__)
_PARAMS = os.path.normpath(os.path.join(_HERE, "..", "data", "parameters.json"))


class ComponentModel:
    component_id = "EC198"
    component_name = "Post-Combustion Capture (Amine Scrubbing)"
    fidelity = "F0a -- empirical lookup"
    version = "1.0.0"

    def __init__(self, params_path=_PARAMS):
        with open(params_path) as fh:
            self.params = json.load(fh)
        self.model = CaptureEnergyCurve(self.params)

    def predict(self, inputs):
        return self.model.predict(inputs)

    def get_info(self):
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {'flue_gas_rate': 'kg/s', 'co2_fraction': 'mol/mol', 'capture_rate': '-'},
            "outputs": {'co2_captured_kg_s': 'kg/s', 'reboiler_duty_GJ_tCO2': 'GJ/tCO2', 'total_specific_energy_GJ_tCO2': 'GJ/tCO2'},
            "source": self.params.get("source") or self.params.get("unit", {}).get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.component_id, "-", m.component_name, "[", m.fidelity, "]")
    for k, v in m.predict({'flue_gas_rate': 500.0, 'co2_fraction': 0.12, 'capture_rate': 0.9}).items():
        print("  {} = {}".format(k, v))
