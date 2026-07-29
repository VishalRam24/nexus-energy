"""
EC020 -- NCA Battery -- F2a ECM 1-RC -- Standardized Predict Interface
"""

import json
import numpy as np
from pathlib import Path
from model import NCABatteryECM1RC


class ComponentModel:
    """Standardized interface for EC020 NCA Battery -- F2a 1-RC ECM."""

    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = NCABatteryECM1RC(self.params)

    def predict(self, inputs: dict) -> dict:
        current = np.atleast_1d(np.asarray(inputs["current"], dtype=float))
        dt = float(inputs.get("dt", 1.0))
        soc_init = float(inputs.get("soc_init", 1.0))

        self._model.reset(soc_init)
        result = self._model.simulate(current, dt)
        return result

    def get_info(self) -> dict:
        return {
            "name": "NCA Battery (Nickel-Cobalt-Aluminum)",
            "ec_id": "EC020",
            "fidelity": "F2a",
            "description": "1-RC Equivalent Circuit Model: V = OCV(SOC) - I*R0 - V_rc",
            "inputs": {
                "current": {"unit": "A", "note": "positive=discharge, array for time series"},
                "dt": {"unit": "s", "default": 1.0},
                "soc_init": {"unit": "dimensionless", "range": [0.0, 1.0], "default": 1.0},
            },
            "outputs": {
                "voltage": {"unit": "V"},
                "soc": {"unit": "dimensionless"},
                "power": {"unit": "W"},
                "v_rc": {"unit": "V"},
                "time": {"unit": "s"},
            },
            "parameters": {
                "R0": f"{self.params['cell']['R0']['value']} Ohm (nominal)",
                "R1": f"{self.params['cell']['R1']['value']} Ohm (nominal)",
                "C1": f"{self.params['cell']['C1']['value']} F (nominal)",
                "tau1": f"{self.params['cell']['R1']['value'] * self.params['cell']['C1']['value']:.1f} s",
                "Q_nom": f"{self.params['cell']['capacity']['value']} Ah",
            },
            "source": self.params.get("source", ""),
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    print(json.dumps(model.get_info(), indent=2))
    Q = model.params["cell"]["capacity"]["value"]
    result = model.predict({"current": [Q] * 3600, "dt": 1.0, "soc_init": 1.0})
    print(f"\n1C Discharge ({Q:.1f}A): Start V={result['voltage'][0]:.4f}, End V={result['voltage'][-1]:.4f}")
