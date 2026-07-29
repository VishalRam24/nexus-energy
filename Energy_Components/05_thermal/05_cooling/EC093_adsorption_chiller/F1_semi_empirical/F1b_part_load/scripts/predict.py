"""EC093 — Adsorption Chiller — F1b Part-Load — Standardized Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import AdsorptionChillerF1b


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = AdsorptionChillerF1b(self.params)

    def predict(self, inputs: dict) -> dict:
        T_hot     = np.asarray(inputs["T_hot"],     dtype=float)
        T_cool    = np.asarray(inputs["T_cool"],    dtype=float)
        T_chilled = np.asarray(inputs["T_chilled"], dtype=float)
        plr       = np.asarray(inputs.get("part_load_ratio", 1.0), dtype=float)
        return {
            "cop":                  self._model.cop(T_hot, T_cool, T_chilled, plr),
            "cooling_power_kw":     self._model.cooling_power(plr),
            "driving_heat_kw":      self._model.driving_heat(T_hot, T_cool, T_chilled, plr),
            "heat_rejection_kw":    self._model.heat_rejection(T_hot, T_cool, T_chilled, plr),
            "electrical_input_kw":  self._model.electrical_input(plr),
            "cop_degradation_factor": self._model.cop_degradation_factor(plr),
        }

    def get_info(self) -> dict:
        return {
            "name": "Adsorption Chiller (Silica-Gel/Water)",
            "ec_id": "EC093",
            "fidelity": "F1b",
            "description": (
                "Part-load COP with combined PLF = PLF_linear * PLF_kinetic. "
                "PLF_linear = 1-C_d*(1-PLR); PLF_kinetic from half-cycle time shortening. "
                "Cycling losses below PLR_min."
            ),
            "inputs": {
                "T_hot":     {"unit": "degC", "range": [55.0, 95.0]},
                "T_cool":    {"unit": "degC", "range": [22.0, 40.0]},
                "T_chilled": {"unit": "degC", "range": [6.0, 20.0]},
                "part_load_ratio": {"unit": "-", "range": [0.0, 1.0], "default": 1.0},
            },
            "outputs": {
                "cop":                  {"unit": "dimensionless"},
                "cooling_power_kw":     {"unit": "kW_th"},
                "driving_heat_kw":      {"unit": "kW_th"},
                "heat_rejection_kw":    {"unit": "kW_th"},
                "electrical_input_kw":  {"unit": "kW_e"},
                "cop_degradation_factor": {"unit": "dimensionless"},
            },
            "source": "Saha et al. (1995); Wang & Oliveira (2006); Duong et al. (2018)",
        }


if __name__ == "__main__":
    model = ComponentModel()
    print("Part-load sweep (T_hot=85C, T_cool=30C, T_chilled=14C):")
    for plr in [1.0, 0.75, 0.5, 0.35, 0.2, 0.1]:
        r = model.predict({"T_hot": 85.0, "T_cool": 30.0, "T_chilled": 14.0,
                           "part_load_ratio": plr})
        print(f"  PLR={plr:.2f}: COP={float(r['cop']):.3f}, "
              f"Q_cool={float(r['cooling_power_kw']):.1f}kW, "
              f"degradation={float(r['cop_degradation_factor']):.3f}")
