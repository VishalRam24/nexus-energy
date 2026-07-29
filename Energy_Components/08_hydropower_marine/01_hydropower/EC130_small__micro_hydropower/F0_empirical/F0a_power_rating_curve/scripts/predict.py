"""EC130 — Small/Micro Hydropower — F0a — standardized predict interface."""
import json, numpy as np
from pathlib import Path
from model import HydroF0a


class ComponentModel:
    component_id = "EC130"
    component_name = "Small/Micro Hydropower"
    fidelity = "F0a — empirical lookup"
    version = "1.0.0"

    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = HydroF0a(self.params)

    def predict(self, inputs: dict) -> dict:
        Q = np.asarray(inputs["flow_rate_m3s"], dtype=float)
        H = np.asarray(inputs.get("head_m", self._model.H_design), dtype=float)
        return {
            "power_kw": self._model.power_kw(Q, H),
            "overall_efficiency": self._model.overall_efficiency(Q),
            "capacity_factor": self._model.capacity_factor(Q, H),
        }

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "flow_rate_m3s": {"unit": "m3/s", "range": [0.0, 1.65]},
                "head_m": {"unit": "m", "range": [2.0, 1800.0]},
            },
            "outputs": {
                "power_kw": {"unit": "kW"},
                "overall_efficiency": {"unit": "dimensionless"},
                "capacity_factor": {"unit": "dimensionless"},
            },
            "source": self.params["source"],
        }


if __name__ == "__main__":
    m = ComponentModel()
    r = m.predict({"flow_rate_m3s": 1.5, "head_m": 40.0})
    print(f"At design: P={float(r['power_kw']):.0f} kW, eta={float(r['overall_efficiency']):.3f}, CF={float(r['capacity_factor']):.3f}")
