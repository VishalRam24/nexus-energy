"""EC048 — Perovskite Solar Cell — F1a Single-Diode — Standardized Predict Interface"""

import json, numpy as np
from pathlib import Path
from model import PerovskitePVF1a


class ComponentModel:
    """Standardized interface for EC048 Perovskite Solar Cell — F1a single-diode model."""

    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = PerovskitePVF1a(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        Args:
            inputs: {
                "irradiance":       W/m2  (0-1200),
                "cell_temperature": degC  (-10 to 80)
            }
        Returns:
            dict with v_mp [V], i_mp [A], p_mp [W], v_oc [V], i_sc [A], efficiency [-]
        """
        G = np.asarray(inputs["irradiance"], dtype=float)
        T = np.asarray(inputs["cell_temperature"], dtype=float)
        result = self._model.mpp(G, T)
        result["efficiency"] = self._model.efficiency(G, T)
        return result

    def get_info(self) -> dict:
        return {
            "name": "Perovskite Solar Cell",
            "ec_id": "EC048",
            "fidelity": "F1a",
            "description": (
                "Single-diode model (De Soto 5-parameter) for lab-scale MAPbI3 perovskite cell. "
                "Eg=1.55 eV, n=1.5, area=25 cm2. Uses pvlib singlediode solver when available."
            ),
            "inputs": {
                "irradiance":       {"unit": "W/m2", "range": [0.0, 1200.0]},
                "cell_temperature": {"unit": "degC",  "range": [-10.0, 80.0]},
            },
            "outputs": {
                "v_mp":       {"unit": "V"},
                "i_mp":       {"unit": "A"},
                "p_mp":       {"unit": "W"},
                "v_oc":       {"unit": "V"},
                "i_sc":       {"unit": "A"},
                "efficiency": {"unit": "dimensionless"},
            },
            "source": (
                "De Soto et al. (2006), Solar Energy 80(1); "
                "Miyano et al. (2016), J. Phys. Chem. Lett. 7; "
                "NREL Efficiency Chart 2024; pvlib BSD-3"
            ),
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    print(json.dumps(model.get_info(), indent=2))
    r = model.predict({"irradiance": 1000.0, "cell_temperature": 25.0})
    print("\nAt STC (1000 W/m2, 25C):")
    for k, v in r.items():
        print(f"  {k}: {float(v):.4f}")
