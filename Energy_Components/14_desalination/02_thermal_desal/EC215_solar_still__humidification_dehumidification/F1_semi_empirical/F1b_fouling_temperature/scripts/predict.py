"""EC215 — Solar Still / HDH — F1b GOR + Solar + Temperature — Standardized Predict Interface"""
import json
import numpy as np
from pathlib import Path
from model import HDHF1b


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = HDHF1b(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        Predict HDH performance.

        Parameters
        ----------
        inputs : dict
            T_top_degC  : float (degC)      default 70.0
            G_Wm2       : float (W/m2)      default 700.0
            T_amb_degC  : float (degC)      default 25.0
            Lambda      : float (kg/kg)     default None (optimal)
            T_cond_degC : float (degC)      default None (parameter value)
        """
        T_top = np.asarray(inputs.get("T_top_degC", 70.0), dtype=float)
        G     = inputs.get("G_Wm2", 700.0)
        T_a   = inputs.get("T_amb_degC", 25.0)
        Lam   = inputs.get("Lambda", None)
        T_c   = inputs.get("T_cond_degC", None)

        return self._model.compute(T_top, G, T_a, Lam, T_c)

    def get_info(self) -> dict:
        return {
            "name": "Solar Still / Humidification-Dehumidification (HDH)",
            "ec_id": "EC215",
            "fidelity": "F1b",
            "description": (
                "HDH model with GOR vs top brine temperature, psychrometric humidity ratio, "
                "Hottel-Whillier solar collector efficiency, and air-to-water ratio optimization."
            ),
            "inputs": {
                "T_top_degC":  {"unit": "degC",        "range": [50, 85]},
                "G_Wm2":       {"unit": "W/m2",        "range": [100, 1000]},
                "T_amb_degC":  {"unit": "degC",        "range": [10, 45]},
                "Lambda":      {"unit": "kg_air/kg_water", "range": [0.5, 5.0]},
                "T_cond_degC": {"unit": "degC",        "range": [20, 45]},
            },
            "outputs": {
                "gor":             {"unit": "dimensionless"},
                "distillate_kg_h": {"unit": "kg/h"},
                "sec_kwh_m3":      {"unit": "kWh_th/m3"},
                "solar_heat_kw":   {"unit": "kW"},
                "humidity_diff":   {"unit": "kg_water/kg_air"},
            },
            "source": "Narayan et al. (2010); Hermosillo et al. (2012); Müller-Holst et al. (1998)",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    r = model.predict({"T_top_degC": 70.0, "G_Wm2": 700.0, "T_amb_degC": 25.0})
    print("Design point (T_top=70C, G=700 W/m2, T_amb=25C):")
    for k, v in r.items():
        print(f"  {k} = {float(np.atleast_1d(v)[0]):.4f}")
