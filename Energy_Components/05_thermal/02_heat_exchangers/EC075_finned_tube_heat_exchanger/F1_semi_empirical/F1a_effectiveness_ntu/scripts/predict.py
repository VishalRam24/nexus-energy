"""EC075 — Finned-Tube HX — F1a e-NTU — Standardized Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import FinnedTubeHXF1a


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = FinnedTubeHXF1a(self.params)

    def predict(self, inputs: dict) -> dict:
        return self._model.predict(
            T_h_in=np.asarray(inputs["T_h_in"], dtype=float),
            T_c_in=np.asarray(inputs["T_c_in"], dtype=float),
            m_dot_hot=np.asarray(inputs["m_dot_hot"], dtype=float),
            m_dot_cold=np.asarray(inputs["m_dot_cold"], dtype=float),
        )

    def get_info(self) -> dict:
        return {
            "name": "Finned-Tube Heat Exchanger",
            "ec_id": "EC075",
            "fidelity": "F1a",
            "description": "Cross-flow e-NTU with constant U",
            "inputs": {
                "T_h_in": {"unit": "degC", "range": [20.0, 120.0]},
                "T_c_in": {"unit": "degC", "range": [-20.0, 50.0]},
                "m_dot_hot": {"unit": "kg/s", "range": [0.05, 5.0]},
                "m_dot_cold": {"unit": "kg/s", "range": [0.1, 10.0]},
            },
            "outputs": {
                "Q_kw": {"unit": "kW"},
                "T_h_out": {"unit": "degC"},
                "T_c_out": {"unit": "degC"},
                "effectiveness": {"unit": "dimensionless"},
                "NTU": {"unit": "dimensionless"},
            },
            "source": "Incropera & DeWitt (2006); Kays & London (1984)",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    m = ComponentModel()
    r = m.predict({"T_h_in": 80.0, "T_c_in": 20.0, "m_dot_hot": 1.0, "m_dot_cold": 2.0})
    print(f"Q={float(r['Q_kw']):.1f} kW, eps={float(r['effectiveness']):.3f}")
