"""EC048 — Perovskite Solar Cell — F1b Two-Diode + Hysteresis — Standardized Predict Interface"""

import json, numpy as np
from pathlib import Path
from model import PerovskitePVF1b


class ComponentModel:
    """Standardized interface for EC048 Perovskite PV — F1b two-diode + hysteresis."""

    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = PerovskitePVF1b(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        Args:
            inputs: {
                "irradiance_w_m2": W/m2 (0-1200),
                "temperature_cell_degC": degC (-10 to 80),
                "irradiance_rate_w_m2_s": W/m2/s (optional, default 0)
            }
        Returns:
            i_mp, v_mp, p_mp, i_sc, v_oc, efficiency, hysteresis_index
        """
        G = np.asarray(inputs["irradiance_w_m2"], dtype=float)
        T = np.asarray(inputs["temperature_cell_degC"], dtype=float)
        dGdt = np.asarray(inputs.get("irradiance_rate_w_m2_s", 0.0), dtype=float)
        return self._model.mpp(G, T, dGdt)

    def get_info(self) -> dict:
        return {
            "name": "Perovskite Solar Cell",
            "ec_id": "EC048",
            "fidelity": "F1b",
            "description": "Two-diode model with hysteresis correction for ion-migration effects. P_actual = P_two_diode * (1 - h_factor*|dG/dt|/G_ref).",
            "inputs": {
                "irradiance_w_m2": {"unit": "W/m2", "range": [0.0, 1200.0]},
                "temperature_cell_degC": {"unit": "degC", "range": [-10.0, 80.0]},
                "irradiance_rate_w_m2_s": {"unit": "W/m2/s", "range": [-1000.0, 1000.0], "default": 0.0},
            },
            "outputs": {
                "i_mp": {"unit": "A"}, "v_mp": {"unit": "V"}, "p_mp": {"unit": "W"},
                "i_sc": {"unit": "A"}, "v_oc": {"unit": "V"},
                "efficiency": {"unit": "dimensionless"},
                "hysteresis_index": {"unit": "dimensionless"},
            },
            "source": "Tress (2017), J. Phys. Chem. Lett. 8, 3106; Miyano et al. (2016)",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    print(json.dumps(model.get_info(), indent=2))
    r = model.predict({"irradiance_w_m2": 1000.0, "temperature_cell_degC": 25.0})
    print(f"\nAt STC (1000 W/m2, 25C, steady):")
    for k, v in r.items():
        print(f"  {k}: {float(v):.6f}")
