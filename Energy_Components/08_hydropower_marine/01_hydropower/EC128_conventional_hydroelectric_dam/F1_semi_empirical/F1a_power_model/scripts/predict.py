"""EC128 — Hydroelectric Dam — F1a Power Model — Standardized Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import HydroelectricDamF1a


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = HydroelectricDamF1a(self.params)

    def predict(self, inputs: dict) -> dict:
        Q = np.asarray(inputs["flow_rate_m3s"], dtype=float)
        H = np.asarray(inputs["head_m"], dtype=float)
        return {
            "power_kw": self._model.power_kw(Q, H),
            "turbine_efficiency": self._model.turbine_efficiency(Q),
            "overall_efficiency": self._model.overall_efficiency(Q),
            "capacity_factor": self._model.capacity_factor(Q, H),
        }

    def get_info(self) -> dict:
        return {
            "name": "Conventional Hydroelectric Dam",
            "ec_id": "EC128",
            "fidelity": "F1a",
            "description": "P = eta_overall * rho * g * Q * H / 1000 kW; Francis turbine parabolic eta curve",
            "inputs": {
                "flow_rate_m3s": {"unit": "m3/s", "range": [0.0, 33.0]},
                "head_m": {"unit": "m", "range": [50.0, 150.0]},
            },
            "outputs": {
                "power_kw": {"unit": "kW"},
                "turbine_efficiency": {"unit": "dimensionless"},
                "overall_efficiency": {"unit": "dimensionless"},
                "capacity_factor": {"unit": "dimensionless"},
            },
            "source": "Dixon & Hall (2014), Fluid Mechanics and Thermodynamics of Turbomachinery",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    r = model.predict({"flow_rate_m3s": 30.0, "head_m": 100.0})
    print(f"At design point: P={float(r['power_kw']):.1f} kW, eta_t={float(r['turbine_efficiency']):.3f}, eta_ov={float(r['overall_efficiency']):.3f}, CF={float(r['capacity_factor']):.3f}")
