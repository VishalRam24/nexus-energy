"""EC007 -- Reversible Fuel Cell (RFC) -- F1a Polarization Curve (bidirectional)"""
import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from model import RFCPolarizationModel

_PARAMS = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


class ComponentModel:
    component_id   = "EC007"
    component_name = "Reversible Fuel Cell (RFC)"
    fidelity       = "F1a -- Polarization Curve (bidirectional FC/EL)"
    version        = "1.0.0"

    def __init__(self, params=None):
        with open(_PARAMS) as f:
            self._raw = json.load(f)
        p = self._raw["default_parameters"].copy()
        if params:
            p.update(params)
        self._model = RFCPolarizationModel(p)

    def predict(self, inputs: dict) -> dict:
        return self._model.evaluate(**inputs)

    def get_info(self) -> dict:
        return {
            "component_id":   self.component_id,
            "component_name": self.component_name,
            "fidelity":       self.fidelity,
            "version":        self.version,
            "inputs":  {"j": "current density (A/cm^2), positive=FC, negative=EL"},
            "outputs": {
                "V_cell":     "cell voltage (V)",
                "V_stack":    "stack voltage (V)",
                "mode":       "operating mode: FC, EL, or OCV",
                "P_density":  "power density (W/cm^2), positive=output",
                "P_stack":    "stack power (W)",
                "efficiency": "|V_cell / E_rev|",
            },
        }
