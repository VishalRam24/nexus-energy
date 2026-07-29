"""EC030 -- Nickel-Cadmium (NiCd) Battery -- F1a SOC-only"""
import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from model import NiCdBatteryModel

_PARAMS = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


class ComponentModel:
    component_id   = "EC030"
    component_name = "Nickel-Cadmium (NiCd) Battery"
    fidelity       = "F1a -- SOC-only (flat OCV + R_int)"
    version        = "1.0.0"

    def __init__(self, params=None):
        with open(_PARAMS) as f:
            self._raw = json.load(f)
        p = self._raw["default_parameters"].copy()
        if params:
            p.update(params)
        self._model = NiCdBatteryModel(p)

    def predict(self, inputs: dict) -> dict:
        return self._model.evaluate(**inputs)

    def get_info(self) -> dict:
        return {
            "component_id":   self.component_id,
            "component_name": self.component_name,
            "fidelity":       self.fidelity,
            "version":        self.version,
            "inputs":  {
                "soc": "state of charge [0, 1]",
                "I":   "current (A), positive=discharge",
                "dt":  "time step (s)",
            },
            "outputs": {
                "V_terminal": "terminal voltage (V)",
                "OCV":        "open-circuit voltage (V)",
                "SOC_new":    "updated SOC",
                "P":          "power (W)",
                "energy_Wh":  "available energy (Wh)",
            },
        }
