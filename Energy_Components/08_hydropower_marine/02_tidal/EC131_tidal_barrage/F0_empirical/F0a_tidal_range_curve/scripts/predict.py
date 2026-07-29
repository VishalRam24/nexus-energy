"""EC131 — Tidal Barrage — F0a — standardized predict interface."""
import json, numpy as np
from pathlib import Path
from model import TidalRangeF0a


class ComponentModel:
    component_id = "EC131"
    component_name = "Tidal Barrage"
    fidelity = "F0a — empirical lookup"
    version = "1.0.0"

    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = TidalRangeF0a(self.params)

    def predict(self, inputs: dict) -> dict:
        h = np.asarray(inputs["tidal_range_amplitude_m"], dtype=float)
        return {
            "mean_power_mw": self._model.mean_power_mw(h),
            "capacity_factor": self._model.capacity_factor(h),
        }

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "tidal_range_amplitude_m": {"unit": "m", "range": [0.0, 12.0],
                                            "note": "Half peak-to-trough tidal range"},
            },
            "outputs": {
                "mean_power_mw": {"unit": "MW"},
                "capacity_factor": {"unit": "dimensionless"},
            },
            "source": self.params["source"],
        }


if __name__ == "__main__":
    m = ComponentModel()
    r = m.predict({"tidal_range_amplitude_m": 8.0})
    print(f"At design h=8 m: P_avg={float(r['mean_power_mw']):.2f} MW, CF={float(r['capacity_factor']):.3f}")
