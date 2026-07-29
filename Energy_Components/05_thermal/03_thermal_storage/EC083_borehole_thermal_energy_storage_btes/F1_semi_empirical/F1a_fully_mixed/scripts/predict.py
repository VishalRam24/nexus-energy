"""EC083 — BTES — F1a Fully Mixed — Standardized Predict Interface"""
import json, os, sys
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from model import BTESF1a


class ComponentModel:
    component_id = "EC083"

    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = BTESF1a(self.params)

    def predict(self, inputs: dict) -> dict:
        return self._model.predict(
            T_store=np.asarray(inputs["T_store"], dtype=float),
            T_amb=np.asarray(inputs.get("T_amb", 10.0), dtype=float),
            Q_charge=np.asarray(inputs.get("Q_charge", 0.0), dtype=float),
            Q_discharge=np.asarray(inputs.get("Q_discharge", 0.0), dtype=float),
        )

    def get_info(self) -> dict:
        return {
            "name": "Borehole Thermal Energy Storage (BTES)",
            "ec_id": self.component_id,
            "fidelity": "F1a",
            "description": "Fully-mixed: E = m*Cp*dT, Q_loss = UA*(T-T_amb)",
            "inputs": {
                "T_store": {"unit": "degC", "range": [5.0, 95.0]},
                "T_amb": {"unit": "degC", "range": [0.0, 20.0]},
                "Q_charge": {"unit": "W", "range": [0.0, 500000.0]},
                "Q_discharge": {"unit": "W", "range": [0.0, 500000.0]},
            },
            "outputs": {
                "E_stored_MWh": {"unit": "MWh"},
                "Q_loss_kW": {"unit": "kW"},
                "SOC": {"unit": "dimensionless"},
                "Q_net_kW": {"unit": "kW"},
                "E_max_MWh": {"unit": "MWh"},
            },
            "source": "Nordell (1994). Luleå University; Hellström (1991)",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    m = ComponentModel()
    r = m.predict({"T_store": 60.0, "T_amb": 10.0})
    print(f"E={float(r['E_stored_MWh']):.1f} MWh, SOC={float(r['SOC']):.3f}, Q_loss={float(r['Q_loss_kW']):.1f} kW")
