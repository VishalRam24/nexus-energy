"""EC076 — Regenerative Heat Exchanger — F1b Fouling + Carryover — Standardized Predict Interface"""

import json
import numpy as np
from pathlib import Path
from model import RegenerativeHXF1b


class ComponentModel:
    """Standardized interface for EC076 Regenerative HX — F1b fouling + carryover."""

    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = RegenerativeHXF1b(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        Args:
            inputs: {
                "T_h_in": hot-side inlet [degC] (required),
                "T_c_in": cold-side inlet [degC] (required),
                "m_dot_hot": [kg/s] (required),
                "m_dot_cold": [kg/s] (required),
                "fouling_resistance_hot": m2K/W (optional),
                "fouling_resistance_cold": m2K/W (optional),
                "carryover_leakage": 0-0.10 (optional, default from params),
                "Cr_star": dimensionless matrix cap ratio (optional, default from params)
            }
        """
        T_h_in = np.asarray(inputs["T_h_in"], dtype=float)
        T_c_in = np.asarray(inputs["T_c_in"], dtype=float)
        m_dot_h = np.asarray(inputs["m_dot_hot"], dtype=float)
        m_dot_c = np.asarray(inputs["m_dot_cold"], dtype=float)
        Rf_h = inputs.get("fouling_resistance_hot", None)
        Rf_c = inputs.get("fouling_resistance_cold", None)
        carryover = inputs.get("carryover_leakage", None)
        Cr_star = inputs.get("Cr_star", None)

        return self._model.predict(T_h_in, T_c_in, m_dot_h, m_dot_c,
                                    Rf_h, Rf_c, carryover, Cr_star)

    def get_info(self) -> dict:
        return {
            "name": "Regenerative Heat Exchanger",
            "ec_id": "EC076",
            "fidelity": "F1b",
            "description": (
                "Counter-flow e-NTU with fouling (1/U_fouled = 1/U_clean + Rf_hot + Rf_cold), "
                "Cr* matrix correction (eps_regen = eps_cf*(1 - 1/(9*Cr*^1.93))), "
                "and carryover/leakage penalty (Q_actual = Q_ideal*(1 - X_carryover))."
            ),
            "inputs": {
                "T_h_in": {"unit": "degC", "range": [50.0, 600.0]},
                "T_c_in": {"unit": "degC", "range": [-20.0, 100.0]},
                "m_dot_hot": {"unit": "kg/s", "range": [0.1, 20.0]},
                "m_dot_cold": {"unit": "kg/s", "range": [0.1, 20.0]},
                "fouling_resistance_hot": {"unit": "m2K/W", "range": [0.0, 0.01], "default": 0.0002},
                "fouling_resistance_cold": {"unit": "m2K/W", "range": [0.0, 0.01], "default": 0.0002},
                "carryover_leakage": {"unit": "dimensionless", "range": [0.0, 0.10], "default": 0.03},
                "Cr_star": {"unit": "dimensionless", "range": [1.0, 50.0], "default": 5.0},
            },
            "outputs": {
                "Q_kw": {"unit": "kW"},
                "T_h_out": {"unit": "degC"},
                "T_c_out": {"unit": "degC"},
                "effectiveness": {"unit": "dimensionless", "note": "After carryover penalty"},
                "ntu": {"unit": "dimensionless"},
                "U_fouled": {"unit": "W/m2K"},
                "effectiveness_reduction": {"unit": "dimensionless"},
                "carryover_penalty": {"unit": "dimensionless"},
                "cleanliness_factor": {"unit": "dimensionless"},
            },
            "source": (
                "Incropera & DeWitt (2006) ch.11; Shah & Sekulic (2003) ch.5; "
                "Kays & London (1984); ASHRAE HB 2020"
            ),
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    print(json.dumps(model.get_info(), indent=2))
    BASE = {"T_h_in": 250.0, "T_c_in": 20.0, "m_dot_hot": 3.0, "m_dot_cold": 3.0}
    for Rf in [0.0, 0.0002, 0.001, 0.005]:
        r = model.predict({**BASE, "fouling_resistance_hot": Rf, "fouling_resistance_cold": Rf})
        print(f"Rf={Rf:.4f}: Q={float(r['Q_kw']):.1f}kW, "
              f"eps={float(r['effectiveness']):.3f}, "
              f"U_f={float(r['U_fouled']):.1f} W/m2K, "
              f"carryover={float(r['carryover_penalty']):.3f}")
