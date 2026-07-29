"""EC132 — Tidal Stream Turbine — F1a Power Curve — Standardized Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import TidalStreamTurbineF1a


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = TidalStreamTurbineF1a(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        inputs:
            current_speed_ms  : tidal current speed [m/s]
            water_density     : optional, kg/m3 (default 1025)
        """
        v = np.asarray(inputs["current_speed_ms"], dtype=float)
        rho = inputs.get("water_density", None)
        return {
            "power_kw": self._model.power_kw(v, rho),
            "capacity_factor": self._model.capacity_factor(v, rho),
            "power_coefficient": self._model.power_coefficient(v, rho),
        }

    def get_info(self) -> dict:
        t = self.params["turbine"]
        return {
            "name": "Tidal Stream Turbine",
            "ec_id": "EC132",
            "fidelity": "F1a",
            "description": "P = 0.5*rho*A*Cp*v^3*eta [kW]; cut-in/rated/cut-out velocity model; rho_water=1025 kg/m3",
            "inputs": {
                "current_speed_ms": {"unit": "m/s", "range": [0.0, 4.5]},
                "water_density": {"unit": "kg/m3", "range": [1000.0, 1035.0], "note": "Optional"},
            },
            "outputs": {
                "power_kw": {"unit": "kW"},
                "capacity_factor": {"unit": "dimensionless"},
                "power_coefficient": {"unit": "dimensionless"},
            },
            "source": "Fraenkel (2002); IEC TS 62600-200; EMEC (2009)",
        }


if __name__ == "__main__":
    model = ComponentModel()
    for v in [0.5, 1.0, 2.0, 2.5, 3.5, 4.0]:
        r = model.predict({"current_speed_ms": v})
        print(f"v={v:.1f} m/s  P={float(r['power_kw']):7.1f} kW  "
              f"CF={float(r['capacity_factor']):.3f}  Cp={float(r['power_coefficient']):.3f}")
