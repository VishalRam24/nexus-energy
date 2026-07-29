"""EC074 — Plate Heat Exchanger — F1a e-NTU — Standardized Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import PlateHeatExchangerF1a


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = PlateHeatExchangerF1a(self.params)

    def predict(self, inputs: dict) -> dict:
        T_h_in    = np.asarray(inputs["T_h_in"],    dtype=float)
        T_c_in    = np.asarray(inputs["T_c_in"],    dtype=float)
        m_dot_hot  = np.asarray(inputs["m_dot_hot"],  dtype=float)
        m_dot_cold = np.asarray(inputs["m_dot_cold"], dtype=float)
        return self._model.predict(T_h_in, T_c_in, m_dot_hot, m_dot_cold)

    def get_info(self) -> dict:
        return {
            "name": "Plate Heat Exchanger",
            "ec_id": "EC074",
            "fidelity": "F1a",
            "description": "Counter-flow effectiveness-NTU method; Q = eps * C_min * (T_h_in - T_c_in)",
            "inputs": {
                "T_h_in":    {"unit": "degC", "range": [30.0, 120.0]},
                "T_c_in":    {"unit": "degC", "range": [5.0, 60.0]},
                "m_dot_hot":  {"unit": "kg/s", "range": [0.05, 5.0]},
                "m_dot_cold": {"unit": "kg/s", "range": [0.05, 5.0]},
            },
            "outputs": {
                "Q_kw":          {"unit": "kW"},
                "T_h_out":       {"unit": "degC"},
                "T_c_out":       {"unit": "degC"},
                "effectiveness": {"unit": "dimensionless"},
                "ntu":           {"unit": "dimensionless"},
            },
            "source": "Incropera & DeWitt (2006), ch.11; Shah & Sekulic (2003)",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    r = model.predict({"T_h_in": 80.0, "T_c_in": 20.0, "m_dot_hot": 1.0, "m_dot_cold": 1.0})
    print(f"Q={float(r['Q_kw']):.2f} kW, T_h_out={float(r['T_h_out']):.2f}C, "
          f"T_c_out={float(r['T_c_out']):.2f}C, eps={float(r['effectiveness']):.3f}, NTU={float(r['ntu']):.3f}")
