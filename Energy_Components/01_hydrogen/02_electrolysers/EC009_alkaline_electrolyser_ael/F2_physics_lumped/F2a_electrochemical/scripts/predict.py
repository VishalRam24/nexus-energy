"""EC009 -- AEL -- F2a Electrochemical -- Standardized Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import AELF2a


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = AELF2a(self.params)

    def predict(self, inputs: dict) -> dict:
        j = inputs["current_density"]
        T = inputs.get("T_K", 353.0)
        w = inputs.get("koh_wt_pct", 30.0)
        dt = inputs.get("dt", 1.0)
        duration_s = inputs.get("duration_s", 3600.0)
        return self._model.simulate(j, T, w, dt, duration_s)

    def predict_steady_state(self, inputs: dict) -> dict:
        j = inputs["current_density"]
        T = inputs.get("T_K", 353.0)
        w = inputs.get("koh_wt_pct", 30.0)
        return {
            "cell_voltage": float(self._model.cell_voltage(j, T, w)),
            "stack_voltage": float(self._model.stack_voltage(j, T, w)),
            "h2_rate_mol_s": float(self._model.h2_production_rate(j)),
            "efficiency": float(self._model.efficiency(j, T, w)),
            "bubble_coverage": float(self._model.bubble_coverage(j)),
        }

    def get_info(self) -> dict:
        return {
            "name": "Alkaline Electrolyser (AEL)",
            "ec_id": "EC009",
            "fidelity": "F2a",
            "sub_fidelity": "electrochemical",
            "description": (
                "Physics-lumped AEL model: Nernst + Butler-Volmer activation (anode+cathode) "
                "+ ohmic (KOH electrolyte + diaphragm) + bubble coverage correction."
            ),
            "inputs": {
                "current_density": {"unit": "A/m2", "range": [100, 6000]},
                "T_K": {"unit": "K", "range": [323, 363], "default": 353},
                "koh_wt_pct": {"unit": "%", "range": [20, 40], "default": 30},
                "dt": {"unit": "s", "default": 1.0},
                "duration_s": {"unit": "s", "default": 3600},
            },
            "outputs": {
                "t": {"unit": "s"},
                "voltage": {"unit": "V"},
                "h2_production": {"unit": "mol/s"},
                "efficiency": {"unit": "dimensionless"},
                "bubble_coverage": {"unit": "dimensionless"},
            },
            "source": "Ulleberg (2003); Haug et al. (2017)",
        }


if __name__ == "__main__":
    model = ComponentModel()
    r = model.predict_steady_state({"current_density": 3000.0, "T_K": 353.0, "koh_wt_pct": 30.0})
    print(f"Cell voltage: {r['cell_voltage']:.4f} V")
    print(f"Efficiency: {r['efficiency']:.4f}")
    print(f"Bubble coverage: {r['bubble_coverage']:.4f}")
