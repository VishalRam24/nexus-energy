"""EC049 — Multi-Junction CPV — F1b Two-Diode + Thermal — Standardized Predict Interface"""

import json, numpy as np
from pathlib import Path
from model import MJCPVf1b


class ComponentModel:
    """Standardized interface for EC049 Multi-Junction CPV — F1b two-diode + thermal model."""

    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = MJCPVf1b(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        Args:
            inputs: {
                "dni_w_m2": Direct Normal Irradiance on primary optics (W/m2, 0-1100),
                "T_ambient_degC": ambient temperature (degC, -10 to 50)
            }
        Returns:
            i_mp, v_mp, p_mp, i_sc, v_oc, fill_factor, efficiency, T_cell_c, concentration_ratio
        """
        G = np.asarray(inputs["dni_w_m2"], dtype=float)
        T = np.asarray(inputs["T_ambient_degC"], dtype=float)
        result = self._model.mpp(G, T)
        result["efficiency"] = self._model.efficiency(G, T)
        return result

    def get_info(self) -> dict:
        return {
            "name": "Multi-Junction Concentrator PV (CPV)",
            "ec_id": "EC049",
            "fidelity": "F1b",
            "description": (
                "Two-diode model with concentration-ratio scaling of photocurrent, "
                "logarithmic Voc gain, and CPV-adapted Faiman thermal model. "
                "Representative of GaInP/GaInAs/Ge triple-junction cells at ~500x concentration."
            ),
            "inputs": {
                "dni_w_m2": {"unit": "W/m2", "range": [0.0, 1100.0], "note": "DNI on primary optics"},
                "T_ambient_degC": {"unit": "degC", "range": [-10.0, 50.0]},
            },
            "outputs": {
                "i_mp": {"unit": "A"}, "v_mp": {"unit": "V"}, "p_mp": {"unit": "W"},
                "i_sc": {"unit": "A"}, "v_oc": {"unit": "V"},
                "fill_factor": {"unit": "dimensionless"},
                "efficiency": {"unit": "dimensionless"},
                "T_cell_c": {"unit": "degC"},
                "concentration_ratio": {"unit": "suns"},
            },
            "source": "Araki & Yamaguchi (2003); King et al. (2012); Cotal et al. (2009)",
            "library": "NumPy/SciPy (brentq solver)",
        }


if __name__ == "__main__":
    model = ComponentModel()
    print(json.dumps(model.get_info(), indent=2))
    r = model.predict({"dni_w_m2": 1000.0, "T_ambient_degC": 25.0})
    print("\nAt STC (1000 W/m2 DNI, 25C):")
    for k, v in r.items():
        print(f"  {k}: {float(v):.5f}")
