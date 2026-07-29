"""EC044 — Mono-Si PV — F1b Two-Diode — Standardized Predict Interface"""

import json, numpy as np
from pathlib import Path
from model import MonoSiPVF1b


class ComponentModel:
    """Standardized interface for EC044 Mono-Si PV — F1b two-diode model."""

    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = MonoSiPVF1b(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        Args:
            inputs: {
                "irradiance_w_m2": W/m2 (0-1200),
                "temperature_cell_degC": degC (-10 to 80)
            }
        Returns:
            i_mp, v_mp, p_mp, i_sc, v_oc, fill_factor, efficiency
        """
        G = np.asarray(inputs["irradiance_w_m2"], dtype=float)
        T = np.asarray(inputs["temperature_cell_degC"], dtype=float)
        result = self._model.mpp(G, T)
        result["efficiency"] = self._model.efficiency(G, T)
        return result

    def get_info(self) -> dict:
        return {
            "name": "Monocrystalline Silicon PV",
            "ec_id": "EC044",
            "fidelity": "F1b",
            "description": "Two-diode model with separate diffusion (n1=1) and recombination (n2=2) currents. Better low-irradiance accuracy than F1a single-diode.",
            "inputs": {
                "irradiance_w_m2": {"unit": "W/m2", "range": [0.0, 1200.0]},
                "temperature_cell_degC": {"unit": "degC", "range": [-10.0, 80.0]},
            },
            "outputs": {
                "i_mp": {"unit": "A"}, "v_mp": {"unit": "V"}, "p_mp": {"unit": "W"},
                "i_sc": {"unit": "A"}, "v_oc": {"unit": "V"},
                "fill_factor": {"unit": "dimensionless"},
                "efficiency": {"unit": "dimensionless"},
            },
            "source": "Ishaque et al. (2011), Solar Energy 85(9); De Soto et al. (2006)",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    print(json.dumps(model.get_info(), indent=2))
    r = model.predict({"irradiance_w_m2": 1000.0, "temperature_cell_degC": 25.0})
    print(f"\nAt STC (1000 W/m2, 25C):")
    for k, v in r.items():
        print(f"  {k}: {float(v):.4f}")
