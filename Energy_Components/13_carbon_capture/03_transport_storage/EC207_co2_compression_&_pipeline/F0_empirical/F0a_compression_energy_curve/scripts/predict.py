"""F0a predict interface for EC207 CO2 Compression & Pipeline."""
import os, sys, json

sys.path.insert(0, os.path.dirname(__file__))
from model import CompressionEnergyCurve  # noqa: E402

_HERE = os.path.dirname(__file__)
_PARAMS = os.path.normpath(os.path.join(_HERE, "..", "data", "parameters.json"))


class ComponentModel:
    component_id = "EC207"
    component_name = "CO2 Compression & Pipeline"
    fidelity = "F0a -- empirical lookup"
    version = "1.0.0"

    def __init__(self, params_path=_PARAMS):
        with open(params_path) as fh:
            self.params = json.load(fh)
        self.model = CompressionEnergyCurve(self.params)

    def predict(self, inputs):
        return self.model.predict(inputs)

    def get_info(self):
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {'mass_flow': 'kg/s', 'P_outlet': 'bar', 'pipeline_length_km': 'km'},
            "outputs": {'SEC_kWh_tCO2': 'kWh/tCO2', 'compression_power_MW': 'MW', 'pipeline_pressure_drop_bar': 'bar'},
            "source": self.params.get("source") or self.params.get("unit", {}).get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.component_id, "-", m.component_name, "[", m.fidelity, "]")
    for k, v in m.predict({'mass_flow': 100.0, 'P_outlet': 150.0, 'pipeline_length_km': 100.0}).items():
        print("  {} = {}".format(k, v))
