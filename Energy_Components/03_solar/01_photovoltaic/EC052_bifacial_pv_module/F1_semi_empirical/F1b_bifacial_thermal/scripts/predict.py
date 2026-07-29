"""EC052 — Bifacial PV Module — F1b Bifacial + Thermal — Standardized Predict Interface"""

import json
import numpy as np
from pathlib import Path
from model import BifacialPVF1b


class ComponentModel:
    """Standardized interface for EC052 Bifacial PV — F1b bifacial + thermal model."""

    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = BifacialPVF1b(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        Args:
            inputs: {
                "irradiance_front_w_m2": W/m2 (0-1200),
                "T_ambient_degC": degC (-10 to 50),
                "albedo": dimensionless (0-0.9) [optional, default 0.2],
                "irradiance_rear_w_m2": W/m2 [optional direct rear override]
            }
        Returns:
            i_mp, v_mp, p_mp, i_sc, v_oc, fill_factor, efficiency,
            G_eff_w_m2, G_rear_w_m2,
            T_cell_front_c, T_cell_rear_c, T_cell_eff_c
        """
        G_front = np.asarray(inputs["irradiance_front_w_m2"], dtype=float)
        T_amb = np.asarray(inputs["T_ambient_degC"], dtype=float)
        albedo = float(inputs.get("albedo", 0.2))
        G_rear = inputs.get("irradiance_rear_w_m2", None)
        if G_rear is not None:
            G_rear = np.asarray(G_rear, dtype=float)

        result = self._model.mpp(G_front, T_amb, albedo, G_rear)
        result["efficiency"] = np.where(
            G_front > 1.0,
            result["p_mp"] / (np.maximum(G_front, 1.0) * self._model.area),
            0.0,
        )
        return result

    def get_info(self) -> dict:
        return {
            "name": "Bifacial PV Module",
            "ec_id": "EC052",
            "fidelity": "F1b",
            "description": (
                "Bifacial PV with separate front and rear thermal models. "
                "G_eff = G_front + bifaciality * albedo * G_front * rear_view_factor. "
                "Rear cell temperature is slightly lower than front (rear_thermal_factor=0.85) "
                "because it absorbs less irradiance. Effective cell temp is irradiance-weighted average."
            ),
            "inputs": {
                "irradiance_front_w_m2": {"unit": "W/m2", "range": [0.0, 1200.0]},
                "T_ambient_degC": {"unit": "degC", "range": [-10.0, 50.0]},
                "albedo": {"unit": "dimensionless", "range": [0.0, 0.9], "default": 0.2},
                "irradiance_rear_w_m2": {"unit": "W/m2", "note": "Optional direct rear override"},
            },
            "outputs": {
                "i_mp": {"unit": "A"}, "v_mp": {"unit": "V"}, "p_mp": {"unit": "W"},
                "i_sc": {"unit": "A"}, "v_oc": {"unit": "V"},
                "fill_factor": {"unit": "dimensionless"},
                "efficiency": {"unit": "dimensionless"},
                "G_eff_w_m2": {"unit": "W/m2"},
                "G_rear_w_m2": {"unit": "W/m2"},
                "T_cell_front_c": {"unit": "degC"},
                "T_cell_rear_c": {"unit": "degC"},
                "T_cell_eff_c": {"unit": "degC"},
            },
            "source": "Deline et al. (2017) IEEE PVSC; Marion et al. (2017) IEEE JPV 7(6); Cvetkovska et al. (2021)",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    print(json.dumps(model.get_info(), indent=2))
    r = model.predict({"irradiance_front_w_m2": 1000.0, "T_ambient_degC": 25.0, "albedo": 0.25})
    print(f"\nAt 1000 W/m2 front, T_amb=25C, albedo=0.25:")
    for k, v in r.items():
        print(f"  {k}: {float(np.asarray(v)):.4f}")
