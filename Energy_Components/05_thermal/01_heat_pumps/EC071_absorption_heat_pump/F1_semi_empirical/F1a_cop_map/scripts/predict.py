"""EC071 — Absorption Heat Pump — F1a COP Map — Standardized Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import AbsorptionHeatPumpF1a


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = AbsorptionHeatPumpF1a(self.params)

    def predict(self, inputs: dict) -> dict:
        Tg = np.asarray(inputs["T_gen"],  dtype=float)
        Te = np.asarray(inputs["T_evap"], dtype=float)
        Tc = np.asarray(inputs["T_cond"], dtype=float)
        plr = np.asarray(inputs.get("part_load_ratio", 1.0), dtype=float)
        return {
            "cop":                  self._model.cop(Tg, Te, Tc),
            "heating_capacity_kw":  self._model.heating_capacity(Tg, Te, Tc, plr),
            "driving_heat_kw":      self._model.driving_heat(Tg, Te, Tc, plr),
            "evaporator_heat_kw":   self._model.evaporator_heat(Tg, Te, Tc, plr),
            "electrical_input_kw":  self._model.electrical_input(plr),
        }

    def get_info(self) -> dict:
        return {
            "name": "Absorption Heat Pump (single-effect, LiBr-H2O)",
            "ec_id": "EC071",
            "fidelity": "F1a",
            "description": "Thermally driven Type-I AHP; COP = eta_rev * (eta_engine * COP_carnot_hp + 1)",
            "inputs": {
                "T_gen":  {"unit": "degC", "range": [70.0, 110.0]},
                "T_evap": {"unit": "degC", "range": [0.0, 25.0]},
                "T_cond": {"unit": "degC", "range": [25.0, 50.0]},
                "part_load_ratio": {"unit": "-", "range": [0.0, 1.0], "default": 1.0},
            },
            "outputs": {
                "cop":                 {"unit": "dimensionless"},
                "heating_capacity_kw": {"unit": "kW_th"},
                "driving_heat_kw":     {"unit": "kW_th"},
                "evaporator_heat_kw":  {"unit": "kW_th"},
                "electrical_input_kw": {"unit": "kW_e"},
            },
            "source": "Herold, Radermacher & Klein (2016); Hellmann & Ziegler (1999)",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    r = model.predict({"T_gen": 90.0, "T_evap": 10.0, "T_cond": 35.0})
    print(f"At T_gen=90C, T_evap=10C, T_cond=35C: "
          f"COP={float(r['cop']):.2f}, Q_h={float(r['heating_capacity_kw']):.1f}kW, "
          f"Q_gen={float(r['driving_heat_kw']):.1f}kW, Q_evap={float(r['evaporator_heat_kw']):.1f}kW")
