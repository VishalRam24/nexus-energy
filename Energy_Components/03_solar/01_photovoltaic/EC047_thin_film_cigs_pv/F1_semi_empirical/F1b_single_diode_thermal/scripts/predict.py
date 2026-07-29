"""EC047 — Thin-Film CIGS PV — F1b Single-Diode + Thermal — Standardized Predict Interface"""

import json
import numpy as np
from pathlib import Path
from model import CIGSPvF1b


class ComponentModel:
    """Standardized interface for EC047 CIGS PV — F1b single-diode + thermal model."""

    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = CIGSPvF1b(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        Args:
            inputs: {
                "irradiance_w_m2": W/m2 (0-1200),
                "T_ambient_degC": degC
                  --- OR ---
                "temperature_cell_degC": degC  [optional override]
            }
        Returns:
            i_mp, v_mp, p_mp, i_sc, v_oc, fill_factor, efficiency, T_cell_c
        """
        G = np.asarray(inputs["irradiance_w_m2"], dtype=float)
        if "temperature_cell_degC" in inputs:
            T_cell = np.asarray(inputs["temperature_cell_degC"], dtype=float)
            result = self._model.mpp_from_cell_temp(G, T_cell)
            result["T_cell_c"] = T_cell
        else:
            T_amb = np.asarray(inputs["T_ambient_degC"], dtype=float)
            result = self._model.mpp(G, T_amb)
        result["efficiency"] = np.where(
            G > 1.0,
            result["p_mp"] / (np.maximum(G, 1.0) * self._model.area),
            0.0,
        )
        return result

    def get_info(self) -> dict:
        return {
            "name": "Thin-Film CIGS PV",
            "ec_id": "EC047",
            "fidelity": "F1b",
            "description": (
                "De Soto 5-parameter single-diode + Faiman NOCT cell temperature model. "
                "CIGS tempco ~-0.31 %/K. No additional empirical correction needed "
                "(De Soto captures CIGS thermal behaviour adequately unlike CdTe)."
            ),
            "inputs": {
                "irradiance_w_m2": {"unit": "W/m2", "range": [0.0, 1200.0]},
                "T_ambient_degC": {"unit": "degC", "range": [-10.0, 50.0]},
                "temperature_cell_degC": {"unit": "degC", "range": [-10.0, 90.0],
                                          "note": "Optional direct override"},
            },
            "outputs": {
                "i_mp": {"unit": "A"}, "v_mp": {"unit": "V"}, "p_mp": {"unit": "W"},
                "i_sc": {"unit": "A"}, "v_oc": {"unit": "V"},
                "fill_factor": {"unit": "dimensionless"},
                "efficiency": {"unit": "dimensionless"},
                "T_cell_c": {"unit": "degC"},
            },
            "source": "De Soto (2006); Faiman (2008); Solar Frontier SF170-S datasheet",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    print(json.dumps(model.get_info(), indent=2))
    r = model.predict({"irradiance_w_m2": 1000.0, "T_ambient_degC": 25.0})
    print(f"\nAt 1000 W/m2, T_amb=25C:")
    for k, v in r.items():
        print(f"  {k}: {float(np.asarray(v)):.4f}")
