"""F0a predict interface for EC205 CO2 Electrolyzer (CO2 to CO/Fuels)."""
import os, sys, json

sys.path.insert(0, os.path.dirname(__file__))
from model import SECCurrentCurve  # noqa: E402

_HERE = os.path.dirname(__file__)
_PARAMS = os.path.normpath(os.path.join(_HERE, "..", "data", "parameters.json"))


class ComponentModel:
    component_id = "EC205"
    component_name = "CO2 Electrolyzer (CO2 to CO/Fuels)"
    fidelity = "F0a -- empirical lookup"
    version = "1.0.0"

    def __init__(self, params_path=_PARAMS):
        with open(params_path) as fh:
            self.params = json.load(fh)
        self.model = SECCurrentCurve(self.params)

    def predict(self, inputs):
        return self.model.predict(inputs)

    def get_info(self):
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {'co2_in_kg_s': 'kg/s', 'current_density_mA_cm2': 'mA/cm2'},
            "outputs": {'faradaic_efficiency': '-', 'co_produced_kg_s': 'kg/s', 'SEC_kWh_kgCO2': 'kWh/kgCO2'},
            "source": self.params.get("source") or self.params.get("unit", {}).get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.component_id, "-", m.component_name, "[", m.fidelity, "]")
    for k, v in m.predict({'co2_in_kg_s': 1.0, 'current_density_mA_cm2': 200.0}).items():
        print("  {} = {}".format(k, v))
