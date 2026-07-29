"""EC092 — Absorption Chiller — F1a Characteristic Equation — Standardized Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import AbsorptionChillerF1a


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = AbsorptionChillerF1a(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        inputs:
          T_generator  : generator temperature (degC)
          T_condenser  : condenser temperature (degC)
          T_evaporator : evaporator temperature (degC) — used for context/checks
          Q_cool_kw    : optional cooling demand (kW); defaults to rated 500 kW
        returns:
          cop, cooling_kw, heat_input_kw, heat_rejection_kw
        """
        T_gen = np.asarray(inputs["T_generator"], dtype=float)
        T_cond = np.asarray(inputs["T_condenser"], dtype=float)
        Q_cool_kw = inputs.get("Q_cool_kw", None)

        flows = self._model.heat_flows(T_gen, T_cond, Q_cool_kw)
        return {
            "cop": flows["cop"],
            "cooling_kw": flows["Q_cool_kw"],
            "heat_input_kw": flows["Q_generator_kw"],
            "heat_rejection_kw": flows["Q_reject_kw"],
        }

    def get_info(self) -> dict:
        return {
            "name": "Absorption Chiller (Single-Effect LiBr-H2O)",
            "ec_id": "EC092",
            "fidelity": "F1a",
            "description": "COP = COP_max * (1 - exp(-alpha * (T_gen - T_cond) / dT_ref))",
            "inputs": {
                "T_generator":  {"unit": "degC", "range": [70.0, 120.0]},
                "T_condenser":  {"unit": "degC", "range": [25.0, 45.0]},
                "T_evaporator": {"unit": "degC", "range": [4.0, 15.0]},
                "Q_cool_kw":    {"unit": "kW", "range": [0.0, 500.0], "default": 500.0},
            },
            "outputs": {
                "cop":              {"unit": "dimensionless"},
                "cooling_kw":       {"unit": "kW"},
                "heat_input_kw":    {"unit": "kW"},
                "heat_rejection_kw":{"unit": "kW"},
            },
            "source": "Herold et al. (2016); Gordon & Ng (2000)",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    r = model.predict({"T_generator": 90.0, "T_condenser": 35.0, "T_evaporator": 7.0})
    print(f"Rated: COP={float(r['cop']):.3f}, Q_cool={float(r['cooling_kw']):.1f}kW, "
          f"Q_gen={float(r['heat_input_kw']):.1f}kW, Q_rej={float(r['heat_rejection_kw']):.1f}kW")
