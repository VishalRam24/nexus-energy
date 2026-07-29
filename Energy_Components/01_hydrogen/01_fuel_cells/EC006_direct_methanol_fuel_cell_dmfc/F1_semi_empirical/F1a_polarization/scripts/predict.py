"""EC006 -- Direct Methanol Fuel Cell (DMFC) -- F1a Polarization Curve"""
import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from model import DMFCPolarizationModel

_PARAMS = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


class ComponentModel:
    component_id   = "EC006"
    component_name = "Direct Methanol Fuel Cell (DMFC)"
    fidelity       = "F1a -- Polarization Curve (isothermal, methanol crossover OCV)"
    version        = "1.0.0"

    def __init__(self, params=None):
        with open(_PARAMS) as f:
            self._raw = json.load(f)
        p = self._raw["default_parameters"].copy()
        if params:
            p.update(params)
        self._model = DMFCPolarizationModel(p)

    def predict(self, inputs: dict) -> dict:
        return self._model.evaluate(**inputs)

    def get_info(self) -> dict:
        return {
            "component_id":   self.component_id,
            "component_name": self.component_name,
            "fidelity":       self.fidelity,
            "version":        self.version,
            "inputs":  {"j": "current density (A/cm^2)"},
            "outputs": {
                "V_cell":         "single-cell voltage (V)",
                "V_stack":        "stack voltage (V)",
                "P_density":      "power density (W/cm^2)",
                "P_stack":        "stack power (W)",
                "efficiency":     "voltage efficiency vs E_rev (-)",
                "efficiency_ocv": "voltage efficiency vs OCV (-)",
            },
        }
