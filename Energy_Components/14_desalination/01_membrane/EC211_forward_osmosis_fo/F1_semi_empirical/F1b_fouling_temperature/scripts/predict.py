"""EC211 — Forward Osmosis (FO) — F1b Fouling + Temperature — Standardized Predict Interface"""
import json
import numpy as np
from pathlib import Path
from model import FOF1b


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = FOF1b(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        Predict FO performance.

        Parameters
        ----------
        inputs : dict
            pi_draw_bar     : float (bar)    default 60.0
            pi_feed_bar     : float (bar)    default 27.0
            T_feed_degC     : float (degC)   default 25.0
            operating_hours : float (hours)  default 0.0
        """
        pi_D  = inputs.get("pi_draw_bar", 60.0)
        pi_F  = inputs.get("pi_feed_bar", 27.0)
        T     = inputs.get("T_feed_degC", 25.0)
        hours = inputs.get("operating_hours", 0.0)

        return self._model.compute(pi_D, pi_F, T, hours)

    def get_info(self) -> dict:
        return {
            "name": "Forward Osmosis (FO)",
            "ec_id": "EC211",
            "fidelity": "F1b",
            "description": (
                "FO model with draw solution reconcentration energy, internal concentration "
                "polarization (ICP) correction, Arrhenius temperature effect on permeability, "
                "and linear membrane fouling (~8%/yr)."
            ),
            "inputs": {
                "pi_draw_bar":    {"unit": "bar",   "range": [10, 120]},
                "pi_feed_bar":    {"unit": "bar",   "range": [0.5, 40]},
                "T_feed_degC":    {"unit": "degC",  "range": [5, 45]},
                "operating_hours": {"unit": "hours", "range": [0, 87600]},
            },
            "outputs": {
                "permeate_flow_m3_h":  {"unit": "m3/h"},
                "water_flux_lmh":      {"unit": "L/(m2*h)"},
                "sec_total_kwh_m3":    {"unit": "kWh/m3"},
                "salt_leakage_mg_l":   {"unit": "mg/L"},
                "flux_decline_factor": {"unit": "dimensionless"},
            },
            "source": "McGinnis & Elimelech (2008); Cath et al. (2006); Zhao et al. (2012)",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    r = model.predict({"pi_draw_bar": 60.0, "pi_feed_bar": 27.0,
                       "T_feed_degC": 25.0, "operating_hours": 0})
    print("Design point (pi_draw=60 bar, pi_feed=27 bar, T=25C, fresh):")
    for k, v in r.items():
        print(f"  {k} = {float(np.atleast_1d(v)[0]):.4f}")
