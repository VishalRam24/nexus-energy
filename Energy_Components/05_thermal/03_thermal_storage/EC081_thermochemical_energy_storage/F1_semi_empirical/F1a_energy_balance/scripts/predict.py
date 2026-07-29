"""EC081 — Thermochemical Storage — F1a Energy Balance — Standardized Predict Interface"""
import json, os, sys
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from model import ThermochemicalStorageF1a


class ComponentModel:
    component_id = "EC081"

    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = ThermochemicalStorageF1a(self.params)

    def predict(self, inputs: dict) -> dict:
        return self._model.predict(
            x=np.asarray(inputs["x"], dtype=float),
            mode=inputs.get("mode", "discharge"),
        )

    def get_info(self) -> dict:
        return {
            "name": "Thermochemical Energy Storage",
            "ec_id": self.component_id,
            "fidelity": "F1a",
            "description": "E = m * dH_rxn * x; eta_rt=0.70; near-zero standby losses",
            "inputs": {
                "x": {"unit": "dimensionless", "range": [0.0, 1.0]},
                "mode": {"unit": "string", "options": ["charge", "discharge"]},
            },
            "outputs": {
                "E_stored_kWh": {"unit": "kWh"},
                "E_usable_kWh": {"unit": "kWh"},
                "SOC": {"unit": "dimensionless"},
                "E_max_kWh": {"unit": "kWh"},
                "eta_rt": {"unit": "dimensionless"},
                "P_1h_kW": {"unit": "kW"},
            },
            "source": "Pardo et al. (2014). Renew. Sust. Energy Rev.",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    m = ComponentModel()
    r = m.predict({"x": 0.75})
    print(f"E_stored={float(r['E_stored_kWh']):.1f} kWh, E_usable={float(r['E_usable_kWh']):.1f} kWh")
