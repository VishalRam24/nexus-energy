"""EC065 — Offshore Fixed-Bottom Wind Turbine — F1a Power Curve — Standardized Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import OffshoreWindF1a


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = OffshoreWindF1a(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        inputs:
            wind_speed (m/s):   Hub-height wind speed [0–30]
            air_density (kg/m3): Air density [0.9–1.4], default 1.225

        returns:
            power_kw (kW)
            capacity_factor (-)
            power_coefficient (-)
        """
        v = np.asarray(inputs["wind_speed"], dtype=float)
        rho = np.asarray(inputs.get("air_density", 1.225), dtype=float)
        return {
            "power_kw": self._model.power(v, rho),
            "capacity_factor": self._model.capacity_factor(v, rho),
            "power_coefficient": self._model.power_coefficient(v, rho),
        }

    def get_info(self) -> dict:
        return {
            "name": "Offshore Fixed-Bottom Wind Turbine (Siemens SWT-3.6-120)",
            "ec_id": "EC065",
            "fidelity": "F1a",
            "description": (
                "Interpolated power curve P(v) with linear air density correction: "
                "P(v, rho) = P_curve(v) * (rho / rho_ref)"
            ),
            "inputs": {
                "wind_speed": {"unit": "m/s", "range": [0.0, 30.0]},
                "air_density": {"unit": "kg/m3", "range": [0.9, 1.4], "default": 1.225},
            },
            "outputs": {
                "power_kw": {"unit": "kW"},
                "capacity_factor": {"unit": "dimensionless"},
                "power_coefficient": {"unit": "dimensionless"},
            },
            "source": "IEC 61400-12-1; windpowerlib; Siemens SWT-3.6-120 datasheet",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    for v in [3.5, 5, 8, 12.5, 15, 20, 25, 26]:
        r = model.predict({"wind_speed": float(v)})
        print(f"v={v:5.1f} m/s: P={float(r['power_kw']):6.0f}kW, "
              f"CF={float(r['capacity_factor']):.3f}, "
              f"Cp={float(r['power_coefficient']):.4f}")
