"""EC093 — Adsorption Chiller — F1a COP Map — Standardized Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import AdsorptionChillerF1a


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = AdsorptionChillerF1a(self.params)

    def predict(self, inputs: dict) -> dict:
        Th  = np.asarray(inputs["T_hot"],     dtype=float)
        Tc  = np.asarray(inputs["T_cool"],    dtype=float)
        Tx  = np.asarray(inputs["T_chilled"], dtype=float)
        plr = np.asarray(inputs.get("part_load_ratio", 1.0), dtype=float)
        return {
            "cop":               self._model.cop(Th, Tc, Tx),
            "cooling_kw":        self._model.cooling_power(plr),
            "driving_heat_kw":   self._model.driving_heat(Th, Tc, Tx, plr),
            "heat_rejection_kw": self._model.heat_rejection(Th, Tc, Tx, plr),
            "electrical_kw":     self._model.electrical_input(plr),
        }

    def get_info(self) -> dict:
        return {
            "name": "Adsorption Chiller (silica-gel/water, single-stage)",
            "ec_id": "EC093",
            "fidelity": "F1a",
            "description": "Cooling COP = eta_rev * eta_engine(T_hot, T_cool) * COP_carnot(T_chilled, T_cool)",
            "inputs": {
                "T_hot":     {"unit": "degC", "range": [55.0, 95.0]},
                "T_cool":    {"unit": "degC", "range": [22.0, 40.0]},
                "T_chilled": {"unit": "degC", "range": [6.0, 20.0]},
                "part_load_ratio": {"unit": "-", "range": [0.0, 1.0], "default": 1.0},
            },
            "outputs": {
                "cop":               {"unit": "dimensionless"},
                "cooling_kw":        {"unit": "kW_th"},
                "driving_heat_kw":   {"unit": "kW_th"},
                "heat_rejection_kw": {"unit": "kW_th"},
                "electrical_kw":     {"unit": "kW_e"},
            },
            "source": "Saha et al. (1995); Wang & Oliveira (2006); Herold et al. (2016)",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    r = model.predict({"T_hot": 85.0, "T_cool": 30.0, "T_chilled": 14.0})
    print(f"At T_hot=85, T_cool=30, T_chw=14: COP_c={float(r['cop']):.2f}, "
          f"Q_cool={float(r['cooling_kw']):.1f}kW, Q_drive={float(r['driving_heat_kw']):.1f}kW, "
          f"Q_rej={float(r['heat_rejection_kw']):.1f}kW")
