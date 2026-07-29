"""EC075 — Finned-Tube Heat Exchanger — F1b Fouling + Property Corrections — Standardized Predict Interface"""

import json
import numpy as np
from pathlib import Path
from model import FinnedTubeHXF1b


class ComponentModel:
    """Standardized interface for EC075 Finned-Tube HX — F1b fouling + property corrections."""

    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = FinnedTubeHXF1b(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        Args:
            inputs: {
                "T_h_in": hot-side inlet temperature [degC] (required),
                "T_c_in": cold-side (air) inlet temperature [degC] (required),
                "m_dot_hot": hot-side mass flow rate [kg/s] (required),
                "m_dot_cold": air-side mass flow rate [kg/s] (required),
                "fouling_resistance_tube": m2K/W (optional, default from params),
                "fouling_resistance_air": m2K/W (optional, default from params)
            }
        Returns:
            {
                "Q_kw": kW,
                "T_h_out": degC,
                "T_c_out": degC,
                "effectiveness": –,
                "ntu": –,
                "U_fouled": W/m2K,
                "U_effective_clean": W/m2K,
                "effectiveness_reduction": –,
                "cleanliness_factor": –
            }
        """
        T_h_in = np.asarray(inputs["T_h_in"], dtype=float)
        T_c_in = np.asarray(inputs["T_c_in"], dtype=float)
        m_dot_h = np.asarray(inputs["m_dot_hot"], dtype=float)
        m_dot_c = np.asarray(inputs["m_dot_cold"], dtype=float)
        Rf_t = inputs.get("fouling_resistance_tube", None)
        Rf_a = inputs.get("fouling_resistance_air", None)

        return self._model.predict(T_h_in, T_c_in, m_dot_h, m_dot_c, Rf_t, Rf_a)

    def get_info(self) -> dict:
        return {
            "name": "Finned-Tube Heat Exchanger",
            "ec_id": "EC075",
            "fidelity": "F1b",
            "description": (
                "Cross-flow e-NTU with fouling: 1/U_fouled = 1/(eta_o*U_eff) + Rf_air/eta_o + Rf_tube. "
                "Property corrections: U_eff ~ (m_dot/m_ref)^0.6 * (mu_bulk/mu_wall)^0.14 * (Pr/Pr_ref)^0.33."
            ),
            "inputs": {
                "T_h_in": {"unit": "degC", "range": [30.0, 120.0]},
                "T_c_in": {"unit": "degC", "range": [-10.0, 40.0]},
                "m_dot_hot": {"unit": "kg/s", "range": [0.05, 10.0]},
                "m_dot_cold": {"unit": "kg/s", "range": [0.1, 20.0]},
                "fouling_resistance_tube": {"unit": "m2K/W", "range": [0.0, 0.01], "default": 0.0001},
                "fouling_resistance_air": {"unit": "m2K/W", "range": [0.0, 0.005], "default": 0.0002},
            },
            "outputs": {
                "Q_kw": {"unit": "kW"},
                "T_h_out": {"unit": "degC"},
                "T_c_out": {"unit": "degC"},
                "effectiveness": {"unit": "dimensionless"},
                "ntu": {"unit": "dimensionless"},
                "U_fouled": {"unit": "W/m2K"},
                "U_effective_clean": {"unit": "W/m2K"},
                "effectiveness_reduction": {"unit": "dimensionless"},
                "cleanliness_factor": {"unit": "dimensionless"},
            },
            "source": (
                "Incropera & DeWitt (2006) ch.11; Kays & London (1984); "
                "TEMA Standards 10th ed.; Sieder & Tate (1936)"
            ),
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    print(json.dumps(model.get_info(), indent=2))
    BASE = {"T_h_in": 70.0, "T_c_in": 20.0, "m_dot_hot": 2.0, "m_dot_cold": 5.0}
    for Rf in [0.0, 0.0001, 0.0005, 0.001]:
        r = model.predict({**BASE, "fouling_resistance_tube": Rf, "fouling_resistance_air": Rf})
        print(f"Rf={Rf:.4f}: Q={float(r['Q_kw']):.2f}kW, "
              f"eps={float(r['effectiveness']):.3f}, "
              f"U_f={float(r['U_fouled']):.1f} W/m2K, "
              f"CF={float(r['cleanliness_factor']):.3f}")
