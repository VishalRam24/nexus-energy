"""EC091 — Vapor Compression Chiller — F1b Part-Load — Standardized Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import ChillerF1b


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = ChillerF1b(self.params)

    def predict(self, inputs: dict) -> dict:
        T_chw = np.asarray(inputs["T_chw"], dtype=float)
        T_cw = np.asarray(inputs["T_cw"], dtype=float)
        plr = np.asarray(inputs.get("PLR", 1.0), dtype=float)

        return {
            "cop": self._model.cop(T_chw, T_cw, plr),
            "cooling_capacity_kw": self._model.cooling_capacity(plr),
            "electrical_input_kw": self._model.electrical_input(T_chw, T_cw, plr),
            "iplv": self._model.iplv(),
        }

    def get_info(self) -> dict:
        return {
            "name": "Vapor Compression Chiller",
            "ec_id": "EC091",
            "fidelity": "F1b",
            "description": "DOE-2 IPLV methodology: COP = COP_ref * f_T(T_cw) / EIR_fPLR(PLR)",
            "inputs": {
                "T_chw": {"unit": "degC", "range": [4.0, 12.0]},
                "T_cw": {"unit": "degC", "range": [15.0, 45.0]},
                "PLR": {"unit": "-", "range": [0.0, 1.0], "default": 1.0},
            },
            "outputs": {
                "cop": {"unit": "dimensionless"},
                "cooling_capacity_kw": {"unit": "kW"},
                "electrical_input_kw": {"unit": "kW"},
                "iplv": {"unit": "dimensionless"},
            },
            "source": "AHRI 550/590; DOE-2; EnergyPlus Chiller:Electric:EIR",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    print(f"IPLV = {model._model.iplv():.2f}")
    for plr in [1.0, 0.75, 0.5, 0.25]:
        r = model.predict({"T_chw": 6.7, "T_cw": 29.4, "PLR": plr})
        print(f"PLR={plr:.2f}: COP={float(r['cop']):.2f}, "
              f"Q={float(r['cooling_capacity_kw']):.0f}kW, "
              f"W={float(r['electrical_input_kw']):.1f}kW")
