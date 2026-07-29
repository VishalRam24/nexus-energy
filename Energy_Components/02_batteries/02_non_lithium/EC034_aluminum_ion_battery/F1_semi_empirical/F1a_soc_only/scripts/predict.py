"""EC034 -- Aluminum-Ion Battery -- F1a SOC-only"""
import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from model import AluminumIonBatteryModel

_PARAMS = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


class ComponentModel:
    component_id   = "EC034"
    component_name = "Aluminum-Ion Battery"
    fidelity       = "F1a -- SOC-only (polynomial OCV + R_int)"
    version        = "1.0.0"

    def __init__(self, params=None):
        with open(_PARAMS) as f:
            self._raw = json.load(f)
        p = self._raw["default_parameters"].copy()
        if params:
            p.update(params)
        self._model = AluminumIonBatteryModel(p)

    def predict(self, inputs: dict) -> dict:
        return self._model.evaluate(**inputs)

    def get_info(self) -> dict:
        return {
            "component_id":   self.component_id,
            "component_name": self.component_name,
            "fidelity":       self.fidelity,
            "version":        self.version,
            "inputs":  {"soc": "[0,1]", "I": "A (positive=discharge)", "dt": "s"},
            "outputs": {"V_terminal": "V", "OCV": "V", "SOC_new": "-", "P": "W", "energy_Wh": "Wh"},
        }
