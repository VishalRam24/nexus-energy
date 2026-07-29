"""EC074 — Plate Heat Exchanger — F1b Fouling — Standardized Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import PlateHeatExchangerF1b


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = PlateHeatExchangerF1b(self.params)

    def predict(self, inputs: dict) -> dict:
        T_h_in = np.asarray(inputs["T_h_in"], dtype=float)
        T_c_in = np.asarray(inputs["T_c_in"], dtype=float)
        m_dot_h = np.asarray(inputs["m_dot_hot"], dtype=float)
        m_dot_c = np.asarray(inputs["m_dot_cold"], dtype=float)
        Rf_h = inputs.get("fouling_resistance_hot", None)
        Rf_c = inputs.get("fouling_resistance_cold", None)

        return self._model.predict(T_h_in, T_c_in, m_dot_h, m_dot_c, Rf_h, Rf_c)

    def get_info(self) -> dict:
        return {
            "name": "Plate Heat Exchanger",
            "ec_id": "EC074",
            "fidelity": "F1b",
            "description": "e-NTU with fouling: 1/U_fouled = 1/U_clean + Rf_hot + Rf_cold",
            "inputs": {
                "T_h_in": {"unit": "degC", "range": [30.0, 120.0]},
                "T_c_in": {"unit": "degC", "range": [5.0, 60.0]},
                "m_dot_hot": {"unit": "kg/s", "range": [0.05, 5.0]},
                "m_dot_cold": {"unit": "kg/s", "range": [0.05, 5.0]},
                "fouling_resistance_hot": {"unit": "m2K/W", "range": [0.0, 0.01], "default": 0.0001},
                "fouling_resistance_cold": {"unit": "m2K/W", "range": [0.0, 0.01], "default": 0.0001},
            },
            "outputs": {
                "Q_kw": {"unit": "kW"},
                "T_h_out": {"unit": "degC"},
                "T_c_out": {"unit": "degC"},
                "effectiveness": {"unit": "dimensionless"},
                "ntu": {"unit": "dimensionless"},
                "U_fouled": {"unit": "W/m2K"},
                "effectiveness_reduction": {"unit": "dimensionless"},
            },
            "source": "Incropera & DeWitt (2006) ch.11; TEMA Standards",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    for Rf in [0.0, 0.0001, 0.0005, 0.001, 0.005]:
        r = model.predict({
            "T_h_in": 80.0, "T_c_in": 20.0,
            "m_dot_hot": 1.0, "m_dot_cold": 1.0,
            "fouling_resistance_hot": Rf, "fouling_resistance_cold": Rf,
        })
        print(f"Rf={Rf:.4f}: Q={float(r['Q_kw']):.1f}kW, "
              f"eps={float(r['effectiveness']):.3f}, "
              f"U_f={float(r['U_fouled']):.0f} W/m2K, "
              f"eps_reduction={float(r['effectiveness_reduction']):.1%}")
