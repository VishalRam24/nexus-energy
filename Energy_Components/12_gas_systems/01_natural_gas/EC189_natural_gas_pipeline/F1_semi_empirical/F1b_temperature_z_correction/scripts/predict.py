"""EC189 — Natural Gas Pipeline — F1b Temperature-Z Correction — Predict Interface"""
import json
import numpy as np
from pathlib import Path
from model import NGPipelineF1b


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = NGPipelineF1b(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        Predict pipeline flow with temperature profile and Papay Z-factor.

        Parameters
        ----------
        inputs : dict
            length_km           : float (km)
            diameter_m          : float (m)
            P_in_bar            : float (bar)
            P_out_bar           : float (bar)
            T_in_K              : float (K, default 278.15)
            m_dot_guess_kg_s    : float (kg/s, default 1.0)
        """
        length_km = inputs.get("length_km", 100.0)
        diameter_m = inputs.get("diameter_m", 0.5)
        P_in = inputs.get("P_in_bar", 70.0)
        P_out = inputs.get("P_out_bar", 55.0)
        T_in = inputs.get("T_in_K", None)
        m_guess = inputs.get("m_dot_guess_kg_s", 1.0)
        return self._model.compute(length_km, diameter_m, P_in, P_out, T_in, m_guess)

    def get_info(self) -> dict:
        return {
            "name": "Natural Gas Pipeline",
            "ec_id": "EC189",
            "fidelity": "F1b",
            "description": (
                "Weymouth equation with Coulter-Bardon temperature profile and "
                "Papay Z-factor. K=3.7435e-3 (Menon 2005 SI confirmed)."
            ),
            "inputs": {
                "length_km": {"unit": "km", "range": [1, 1000]},
                "diameter_m": {"unit": "m", "range": [0.1, 1.5]},
                "P_in_bar": {"unit": "bar", "range": [10, 200]},
                "P_out_bar": {"unit": "bar", "range": [10, 200]},
                "T_in_K": {"unit": "K", "range": [240, 330], "default": 278.15},
                "m_dot_guess_kg_s": {"unit": "kg/s", "default": 1.0},
            },
            "outputs": {
                "flow_rate_std_m3_per_day": {"unit": "m3/day"},
                "flow_rate_kg_per_s": {"unit": "kg/s"},
                "T_avg_K": {"unit": "K"},
                "T_out_K": {"unit": "K"},
                "Z_avg": {"unit": "dimensionless"},
            },
            "source": "Menon (2005); Papay (1985); Coulter & Bardon (1979)",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    r = model.predict({"length_km": 100.0, "diameter_m": 0.5,
                        "P_in_bar": 70.0, "P_out_bar": 55.0, "T_in_K": 285.0})
    print("Design point (100 km, D=0.5 m, 70→55 bar, T_in=285 K):")
    for k, v in r.items():
        print(f"  {k} = {float(np.atleast_1d(v)[0]):.4f}")
