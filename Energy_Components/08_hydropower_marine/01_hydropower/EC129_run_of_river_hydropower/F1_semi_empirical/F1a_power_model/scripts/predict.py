"""EC129 — Run-of-River Hydropower — F1a Power Model — Standardized Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import RunOfRiverF1a


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = RunOfRiverF1a(self.params)

    def predict(self, inputs: dict) -> dict:
        Q = np.asarray(inputs["flow_rate_m3s"], dtype=float)
        H = np.asarray(inputs["gross_head_m"], dtype=float)
        return {
            "power_kw": self._model.power_kw(Q, H),
            "net_head_m": self._model.net_head(H),
            "turbine_efficiency": self._model.turbine_efficiency(Q),
            "overall_efficiency": self._model.overall_efficiency(Q),
            "capacity_factor": self._model.capacity_factor(Q, H),
        }

    def get_info(self) -> dict:
        return {
            "name": "Run-of-River Hydropower",
            "ec_id": "EC129",
            "fidelity": "F1a",
            "description": "P = eta_overall * rho * g * Q * H_net / 1000 kW; low-head (2-20 m), no reservoir storage",
            "inputs": {
                "flow_rate_m3s": {"unit": "m3/s", "range": [0.0, 57.5]},
                "gross_head_m": {"unit": "m", "range": [2.0, 20.0]},
            },
            "outputs": {
                "power_kw": {"unit": "kW"},
                "net_head_m": {"unit": "m"},
                "turbine_efficiency": {"unit": "dimensionless"},
                "overall_efficiency": {"unit": "dimensionless"},
                "capacity_factor": {"unit": "dimensionless"},
            },
            "source": "Penche (1998), Layman's Guidebook; Gordon (2001), Can. J. Civ. Eng.",
        }


if __name__ == "__main__":
    model = ComponentModel()
    r = model.predict({"flow_rate_m3s": 50.0, "gross_head_m": 8.0})
    print(f"Design point: P={float(r['power_kw']):.1f} kW, H_net={float(r['net_head_m']):.2f} m, "
          f"eta_t={float(r['turbine_efficiency']):.3f}, CF={float(r['capacity_factor']):.3f}")
