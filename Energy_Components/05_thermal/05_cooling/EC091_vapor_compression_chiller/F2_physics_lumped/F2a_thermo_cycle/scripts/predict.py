"""EC091 — Vapor Compression Chiller — F2a — Standardized Predict Interface"""
import json
from pathlib import Path
from model import ChillerF2a


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = ChillerF2a(self.params)

    def predict(self, inputs: dict) -> dict:
        return self._model.solve_cycle(
            float(inputs["T_evap_degC"]),
            float(inputs["T_cond_degC"]),
        )

    def get_info(self) -> dict:
        return {
            "name": "Vapor Compression Chiller",
            "ec_id": "EC091",
            "fidelity": "F2a",
            "model_type": "Steady-state vapor compression cycle (cooling)",
            "description": "R134a vapor compression cycle — COP_cooling = Q_evap/W_comp",
            "inputs": {
                "T_evap_degC": {"unit": "degC", "range": [-5, 15]},
                "T_cond_degC": {"unit": "degC", "range": [25, 55]},
            },
            "outputs": {
                "cop_cooling": {"unit": "dimensionless"},
                "cooling_capacity_kw": {"unit": "kW"},
                "compressor_kw": {"unit": "kW"},
                "heat_rejection_kw": {"unit": "kW"},
            },
            "refrigerant": "R134a",
            "source": "ASHRAE Handbook; Stoecker & Jones",
        }


if __name__ == "__main__":
    model = ComponentModel()
    r = model.predict({"T_evap_degC": 5.0, "T_cond_degC": 35.0})
    print(f"COP_cool={r['cop_cooling']:.2f}, Q_cool={r['cooling_capacity_kw']:.1f}kW, "
          f"W_comp={r['compressor_kw']:.2f}kW")
