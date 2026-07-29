"""EC077 — Microchannel HX — F1b Fouling + Part-Load LMTD — Predict Interface"""

import json, numpy as np
from pathlib import Path
from model import MicrochannelHXF1b


class ComponentModel:
    """Standardized interface for EC077 Microchannel HX — F1b model."""

    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = MicrochannelHXF1b(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        Args:
            inputs: {
                "T_h_in"      : float or array [degC]
                "T_c_in"      : float or array [degC]
                "m_dot_hot"   : float or array [kg/s]
                "m_dot_cold"  : float or array [kg/s]
                "Rf_hot"      : float or array [m2K/W] (optional)
                "Rf_cold"     : float or array [m2K/W] (optional)
                "PLR"         : float or array [0.5-1.0] (optional, default 1.0)
            }
        Returns: Q_kw, T_h_out, T_c_out, effectiveness, ntu,
                 U_fouled, effectiveness_reduction, F_lmtd
        """
        return self._model.predict(
            inputs["T_h_in"],
            inputs["T_c_in"],
            inputs["m_dot_hot"],
            inputs["m_dot_cold"],
            inputs.get("Rf_hot", None),
            inputs.get("Rf_cold", None),
            inputs.get("PLR", 1.0),
        )

    def get_info(self) -> dict:
        return {
            "name": "Microchannel Heat Exchanger",
            "ec_id": "EC077",
            "fidelity": "F1b",
            "model": "ε-NTU + fouling resistance + part-load LMTD correction",
            "description": (
                "1/U_f = 1/U_clean + Rf_hot + Rf_cold; "
                "NTU = U_f * A * F_LMTD / C_min; "
                "F_LMTD correction for cross-flow and part-load maldistribution"
            ),
            "inputs": {
                "T_h_in":     {"unit": "degC", "range": [30, 120]},
                "T_c_in":     {"unit": "degC", "range": [5, 40]},
                "m_dot_hot":  {"unit": "kg/s", "range": [0.01, 2.0]},
                "m_dot_cold": {"unit": "kg/s", "range": [0.01, 1.0]},
                "Rf_hot":     {"unit": "m2K/W", "range": [0, 0.005], "default": "from params"},
                "Rf_cold":    {"unit": "m2K/W", "range": [0, 0.005], "default": "from params"},
                "PLR":        {"unit": "-", "range": [0.5, 1.0], "default": 1.0},
            },
            "outputs": {
                "Q_kw":                    {"unit": "kW"},
                "T_h_out":                 {"unit": "degC"},
                "T_c_out":                 {"unit": "degC"},
                "effectiveness":           {"unit": "-"},
                "ntu":                     {"unit": "-"},
                "U_fouled":                {"unit": "W/m2K"},
                "effectiveness_reduction": {"unit": "-"},
                "F_lmtd":                  {"unit": "-"},
            },
            "source": "Incropera & DeWitt (2006) Ch.11; TEMA 10th ed.; Kandlikar & Shah (2012)",
        }


if __name__ == "__main__":
    model = ComponentModel()
    print("EC077 F1b — Microchannel HX — Clean vs. Fouled:")
    r_clean = model.predict({"T_h_in": 80.0, "T_c_in": 20.0,
                              "m_dot_hot": 0.5, "m_dot_cold": 0.3,
                              "Rf_hot": 0.0, "Rf_cold": 0.0})
    r_fouled = model.predict({"T_h_in": 80.0, "T_c_in": 20.0,
                               "m_dot_hot": 0.5, "m_dot_cold": 0.3})
    print(f"  Clean Q: {float(r_clean['Q_kw']):.2f} kW, eps: {float(r_clean['effectiveness']):.4f}")
    print(f"  Fouled Q: {float(r_fouled['Q_kw']):.2f} kW, eps: {float(r_fouled['effectiveness']):.4f}")
    print(f"  Reduction: {float(r_fouled['effectiveness_reduction'])*100:.2f}%")
