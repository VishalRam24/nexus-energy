"""EC052 — Bifacial PV Module — F1a Bifacial Gain — Standardized Predict Interface"""

import json, numpy as np
from pathlib import Path
from model import BifacialPVF1a


class ComponentModel:
    """Standardized interface for EC052 Bifacial PV — F1a bifacial-gain single-diode."""

    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = BifacialPVF1a(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        Args:
            inputs: {
                "irradiance_front":  W/m2 on front face (0-1200),
                "cell_temperature":  degC,
                "irradiance_rear":   W/m2 on rear face   (optional),
                "albedo":            ground albedo 0-0.9 (optional, used if rear absent)
            }
        Returns:
            v_mp, i_mp, p_mp, v_oc, i_sc, efficiency, bifacial_gain,
            G_effective, G_rear_used
        """
        G_f = np.asarray(inputs["irradiance_front"], dtype=float)
        T = np.asarray(inputs["cell_temperature"], dtype=float)
        G_r = inputs.get("irradiance_rear", None)
        alb = inputs.get("albedo", None)
        if G_r is not None:
            G_r = np.asarray(G_r, dtype=float)
        if alb is not None:
            alb = np.asarray(alb, dtype=float)

        result = self._model.mpp(G_f, T, G_rear=G_r, albedo=alb)
        result["efficiency"] = self._model.efficiency(G_f, T, G_rear=G_r, albedo=alb)
        result["bifacial_gain"] = self._model.bifacial_gain(G_f, T, G_rear=G_r, albedo=alb)
        return result

    def get_info(self) -> dict:
        return {
            "name": "Bifacial PV Module",
            "ec_id": "EC052",
            "fidelity": "F1a",
            "description": "Bifacial gain (G_eff = G_front + phi*G_rear) + De Soto single-diode",
            "inputs": {
                "irradiance_front": {"unit": "W/m2", "range": [0.0, 1200.0]},
                "cell_temperature": {"unit": "degC", "range": [-10.0, 80.0]},
                "irradiance_rear":  {"unit": "W/m2", "range": [0.0, 600.0], "optional": True},
                "albedo":           {"unit": "dimensionless", "range": [0.0, 0.9], "optional": True},
            },
            "outputs": {
                "v_mp": {"unit": "V"}, "i_mp": {"unit": "A"}, "p_mp": {"unit": "W"},
                "v_oc": {"unit": "V"}, "i_sc": {"unit": "A"},
                "efficiency": {"unit": "dimensionless"},
                "bifacial_gain": {"unit": "dimensionless"},
                "G_effective": {"unit": "W/m2"},
                "G_rear_used": {"unit": "W/m2"},
            },
            "source": "De Soto et al. (2006); Stein et al. (2017) Sandia Bifacial PV review",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    print(json.dumps(model.get_info(), indent=2))
    r = model.predict({"irradiance_front": 1000.0, "cell_temperature": 25.0,
                       "albedo": 0.25})
    print(f"\nAt STC front 1000 W/m2, 25C, albedo=0.25:")
    for k, v in r.items():
        print(f"  {k}: {float(v):.4f}")
