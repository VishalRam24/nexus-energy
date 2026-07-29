"""EC023 -- LMO Lithium Manganese Oxide Battery -- F1a SOC-only"""
import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from model import LMOBatteryModel

_PARAMS = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


class ComponentModel:
    component_id   = "EC023"
    component_name = "LMO Lithium Manganese Oxide Battery"
    fidelity       = "F1a -- SOC-only (polynomial OCV + R_int)"
    version        = "1.0.0"

    def __init__(self, params=None):
        with open(_PARAMS) as f:
            self._raw = json.load(f)
        p = self._raw["default_parameters"].copy()
        if params:
            p.update(params)
        self._model = LMOBatteryModel(p)

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
                "I":   "current (A), positive=discharge (optional, default 0)",
                "dt":  "time step (s) for SOC update (optional, default 0)",
            },
            "outputs": {
                "V_terminal": "terminal voltage (V)",
                "OCV":        "open-circuit voltage (V)",
                "SOC_new":    "updated SOC",
                "P":          "power (W)",
                "energy_Wh":  "available energy (Wh)",
            },
        }
