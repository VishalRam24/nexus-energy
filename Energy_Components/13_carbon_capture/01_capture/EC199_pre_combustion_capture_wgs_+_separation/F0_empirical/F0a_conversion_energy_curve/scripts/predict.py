"""F0a predict interface for EC199 Pre-Combustion Capture (WGS + Separation)."""
import os, sys, json

sys.path.insert(0, os.path.dirname(__file__))
from model import ConversionEnergyCurve  # noqa: E402

_HERE = os.path.dirname(__file__)
_PARAMS = os.path.normpath(os.path.join(_HERE, "..", "data", "parameters.json"))


class ComponentModel:
    component_id = "EC199"
    component_name = "Pre-Combustion Capture (WGS + Separation)"
    fidelity = "F0a -- empirical lookup"
    version = "1.0.0"

    def __init__(self, params_path=_PARAMS):
        with open(params_path) as fh:
            self.params = json.load(fh)
        self.model = ConversionEnergyCurve(self.params)

    def predict(self, inputs):
        return self.model.predict(inputs)

    def get_info(self):
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {'T_WGS_C': 'degC', 'co_flow_kg_s': 'kg/s', 'eta_sep': '-'},
            "outputs": {'wgs_conversion': '-', 'overall_capture_rate': '-', 'total_specific_energy_GJ_tCO2': 'GJ/tCO2'},
            "source": self.params.get("source") or self.params.get("unit", {}).get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.component_id, "-", m.component_name, "[", m.fidelity, "]")
    for k, v in m.predict({'T_WGS_C': 250.0, 'co_flow_kg_s': 10.0}).items():
        print("  {} = {}".format(k, v))
