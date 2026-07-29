"""EC092 — Absorption Chiller — F2a — Standardized Predict Interface"""
import json
from pathlib import Path
from model import AbsorptionChillerF2a


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = AbsorptionChillerF2a(self.params)

    def predict(self, inputs: dict) -> dict:
        return self._model.solve_cycle(
            float(inputs["T_gen_degC"]),
            float(inputs["T_cond_degC"]),
            float(inputs["T_evap_degC"]),
            float(inputs["T_abs_degC"]),
        )

    def get_info(self) -> dict:
        return {
            "name": "Absorption Chiller (LiBr-Water)",
            "ec_id": "EC092",
            "fidelity": "F2a",
            "model_type": "Single-effect LiBr-water absorption cycle",
            "inputs": {
                "T_gen_degC": {"unit": "degC", "range": [75, 120], "description": "Generator (driving heat) temp"},
                "T_cond_degC": {"unit": "degC", "range": [25, 45]},
                "T_evap_degC": {"unit": "degC", "range": [3, 15]},
                "T_abs_degC": {"unit": "degC", "range": [25, 42]},
            },
            "outputs": {
                "cop": {"unit": "dimensionless"},
                "cooling_kw": {"unit": "kW"},
                "heat_input_kw": {"unit": "kW"},
                "pump_power_kw": {"unit": "kW"},
                "solution_flow_kg_s": {"unit": "kg/s"},
            },
            "source": "Herold et al. (2016); Patek & Klomfar (2006)",
        }


if __name__ == "__main__":
    model = ComponentModel()
    r = model.predict({"T_gen_degC": 90.0, "T_cond_degC": 35.0, "T_evap_degC": 7.0, "T_abs_degC": 35.0})
    print(f"COP={r['cop']:.3f}, Q_cool={r['cooling_kw']:.1f}kW, Q_gen={r['heat_input_kw']:.1f}kW, "
          f"W_pump={r['pump_power_kw']:.3f}kW, f={r['circulation_ratio']:.1f}")
