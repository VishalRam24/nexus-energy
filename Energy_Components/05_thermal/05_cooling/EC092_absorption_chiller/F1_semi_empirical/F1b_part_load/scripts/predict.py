"""EC092 — Absorption Chiller — F1b Part-Load — Standardized Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import AbsorptionChillerF1b


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = AbsorptionChillerF1b(self.params)

    def predict(self, inputs: dict) -> dict:
        T_hot = np.asarray(inputs["T_hot"], dtype=float)
        T_cw = np.asarray(inputs["T_cw"], dtype=float)
        T_chw = np.asarray(inputs["T_chw"], dtype=float)
        plr = np.asarray(inputs.get("PLR", 1.0), dtype=float)

        return {
            "cop": self._model.cop(T_hot, T_cw, T_chw, plr),
            "cooling_capacity_kw": self._model.cooling_capacity(plr),
            "heat_input_kw": self._model.heat_input(T_hot, T_cw, T_chw, plr),
        }

    def get_info(self) -> dict:
        return {
            "name": "Absorption Chiller (Single-Effect LiBr-H2O)",
            "ec_id": "EC092",
            "fidelity": "F1b",
            "description": "COP = COP_ref * f_PLR(PLR) * f_Thot(T_hot) with crystallization limit",
            "inputs": {
                "T_hot": {"unit": "degC", "range": [70.0, 120.0]},
                "T_cw": {"unit": "degC", "range": [25.0, 45.0]},
                "T_chw": {"unit": "degC", "range": [4.0, 15.0]},
                "PLR": {"unit": "-", "range": [0.0, 1.0], "default": 1.0},
            },
            "outputs": {
                "cop": {"unit": "dimensionless"},
                "cooling_capacity_kw": {"unit": "kW"},
                "heat_input_kw": {"unit": "kW"},
            },
            "source": "Herold et al. (2016); Gordon & Ng (2000); ASHRAE (2020) Ch.2",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    for plr in [1.0, 0.75, 0.5, 0.25, 0.15]:
        r = model.predict({"T_hot": 90.0, "T_cw": 30.0, "T_chw": 7.0, "PLR": plr})
        print(f"PLR={plr:.2f}: COP={float(r['cop']):.3f}, "
              f"Q_cool={float(r['cooling_capacity_kw']):.0f}kW, "
              f"Q_heat={float(r['heat_input_kw']):.0f}kW")
