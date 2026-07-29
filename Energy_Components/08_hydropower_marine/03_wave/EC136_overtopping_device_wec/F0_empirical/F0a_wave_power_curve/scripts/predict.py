"""EC136 — Overtopping Device WEC — F0a — standardized predict interface."""
import json, numpy as np
from pathlib import Path
from model import WavePowerF0a


class ComponentModel:
    component_id = "EC136"
    component_name = "Overtopping Device WEC"
    fidelity = "F0a — empirical lookup"
    version = "1.0.0"

    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = WavePowerF0a(self.params)

    def predict(self, inputs: dict) -> dict:
        Hs = np.asarray(inputs["Hs_m"], dtype=float)
        return {
            "power_kw": self._model.power_kw(Hs),
            "capacity_factor": self._model.capacity_factor(Hs),
        }

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "Hs_m": {"unit": "m", "range": [0.0, 8.0],
                         "note": f"Significant wave height; lookup at Te={self._model.Te_ref} s"},
            },
            "outputs": {
                "power_kw": {"unit": "kW"},
                "capacity_factor": {"unit": "dimensionless"},
            },
            "source": self.params["source"],
        }


if __name__ == "__main__":
    m = ComponentModel()
    r = m.predict({"Hs_m": 3.0})
    print(f"At Hs=3 m: P={float(r['power_kw']):.1f} kW, CF={float(r['capacity_factor']):.3f}")
