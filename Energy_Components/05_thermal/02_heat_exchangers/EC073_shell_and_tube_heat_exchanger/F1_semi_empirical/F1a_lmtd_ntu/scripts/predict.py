"""EC073 — Shell-and-Tube HX — F1a LMTD/NTU — Standardized Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import ShellAndTubeHEX1a


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = ShellAndTubeHEX1a(self.params)

    def predict(self, inputs: dict) -> dict:
        T_h_in = np.asarray(inputs["T_h_in"], dtype=float)
        T_c_in = np.asarray(inputs["T_c_in"], dtype=float)
        m_h    = np.asarray(inputs["m_dot_hot"],  dtype=float)
        m_c    = np.asarray(inputs["m_dot_cold"], dtype=float)
        return self._model.predict(T_h_in, T_c_in, m_h, m_c)

    def get_info(self) -> dict:
        return {
            "name": "Shell-and-Tube Heat Exchanger (1 shell, 2 tube passes — TEMA E)",
            "ec_id": "EC073",
            "fidelity": "F1a",
            "description": "1-2 shell-and-tube ε-NTU model with LMTD F-correction diagnostics",
            "inputs": {
                "T_h_in":     {"unit": "degC", "range": [30.0, 200.0]},
                "T_c_in":     {"unit": "degC", "range": [5.0, 80.0]},
                "m_dot_hot":  {"unit": "kg/s", "range": [0.1, 50.0]},
                "m_dot_cold": {"unit": "kg/s", "range": [0.1, 50.0]},
            },
            "outputs": {
                "Q_kw":          {"unit": "kW"},
                "T_h_out":       {"unit": "degC"},
                "T_c_out":       {"unit": "degC"},
                "effectiveness": {"unit": "dimensionless"},
                "ntu":           {"unit": "dimensionless"},
                "lmtd":          {"unit": "K"},
                "f_correction":  {"unit": "dimensionless"},
            },
            "source": "Incropera & DeWitt (2006); Bowman, Mueller & Nagle (1940); TEMA",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    r = model.predict({"T_h_in": 90.0, "T_c_in": 25.0, "m_dot_hot": 5.0, "m_dot_cold": 5.0})
    print(f"Q={float(r['Q_kw']):.1f}kW, T_h_out={float(r['T_h_out']):.1f}C, "
          f"T_c_out={float(r['T_c_out']):.1f}C, eps={float(r['effectiveness']):.3f}, "
          f"NTU={float(r['ntu']):.2f}, LMTD={float(r['lmtd']):.2f}K, F={float(r['f_correction']):.3f}")
