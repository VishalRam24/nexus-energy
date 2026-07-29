"""EC073 — Shell-and-Tube HX — F1b Fouling — Standardized Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import ShellTubeHXF1b


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = ShellTubeHXF1b(self.params)

    def predict(self, inputs: dict) -> dict:
        T_h_in   = np.asarray(inputs["T_h_in"],   dtype=float)
        T_c_in   = np.asarray(inputs["T_c_in"],   dtype=float)
        m_dot_h  = np.asarray(inputs["m_dot_hot"], dtype=float)
        m_dot_c  = np.asarray(inputs["m_dot_cold"], dtype=float)
        Rf_shell = inputs.get("Rf_shell", None)
        Rf_tube  = inputs.get("Rf_tube",  None)
        if Rf_shell is not None:
            Rf_shell = np.asarray(Rf_shell, dtype=float)
        if Rf_tube is not None:
            Rf_tube = np.asarray(Rf_tube, dtype=float)
        return self._model.predict(T_h_in, T_c_in, m_dot_h, m_dot_c, Rf_shell, Rf_tube)

    def get_info(self) -> dict:
        return {
            "name": "Shell-and-Tube Heat Exchanger",
            "ec_id": "EC073",
            "fidelity": "F1b",
            "description": (
                "1-shell-pass, 2-tube-pass e-NTU with fouling: "
                "1/UA_eff = 1/UA_0 + Rf_shell + Rf_tube"
            ),
            "inputs": {
                "T_h_in":    {"unit": "degC", "range": [30.0, 200.0]},
                "T_c_in":    {"unit": "degC", "range": [5.0, 80.0]},
                "m_dot_hot": {"unit": "kg/s", "range": [0.1, 50.0]},
                "m_dot_cold":{"unit": "kg/s", "range": [0.1, 50.0]},
                "Rf_shell":  {"unit": "m2K/W", "range": [0.0, 0.005], "default": 0.0002},
                "Rf_tube":   {"unit": "m2K/W", "range": [0.0, 0.005], "default": 0.0002},
            },
            "outputs": {
                "Q_kw":               {"unit": "kW"},
                "T_h_out":            {"unit": "degC"},
                "T_c_out":            {"unit": "degC"},
                "effectiveness":      {"unit": "-"},
                "ntu":                {"unit": "-"},
                "UA_effective":       {"unit": "W/K"},
                "cleanliness_factor": {"unit": "-"},
                "effectiveness_reduction": {"unit": "-"},
            },
            "source": "Incropera & DeWitt (2006); Shah & Sekulic (2003); TEMA 10th ed.",
        }


if __name__ == "__main__":
    model = ComponentModel()
    Rf_vals = [0.0, 0.0001, 0.0002, 0.0005, 0.001, 0.002]
    for Rf in Rf_vals:
        r = model.predict({"T_h_in": 90.0, "T_c_in": 20.0,
                           "m_dot_hot": 5.0, "m_dot_cold": 5.0,
                           "Rf_shell": Rf, "Rf_tube": Rf})
        print(f"Rf={Rf:.4f} m2K/W: Q={float(r['Q_kw']):.1f}kW, "
              f"eps={float(r['effectiveness']):.3f}, "
              f"CF={float(r['cleanliness_factor']):.3f}, "
              f"eps_reduction={float(r['effectiveness_reduction']):.3f}")
