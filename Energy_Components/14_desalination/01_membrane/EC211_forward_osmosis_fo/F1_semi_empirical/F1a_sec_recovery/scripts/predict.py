"""EC211 — Forward Osmosis — F1a — Standardized Predict Interface"""
import json, os, sys
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from model import ForwardOsmosisF1a


class ComponentModel:
    component_id = "EC211"

    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = ForwardOsmosisF1a(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        Inputs:
            capacity_fraction : float or array [0.1, 1.0]
            include_regen     : bool, default True (include draw regeneration energy)
        Returns:
            recovery, rejection, SEC_membrane_kWh_m3, SEC_regen_kWh_m3,
            SEC_total_kWh_m3, permeate_flow_m3_h, concentrate_flow_m3_h, W_elec_kWh_h
        """
        cf = np.asarray(inputs.get("capacity_fraction", 1.0), dtype=float)
        incl = bool(inputs.get("include_regen", True))

        return {
            "recovery": float(self._model.recovery),
            "rejection": float(self._model.rejection),
            "SEC_membrane_kWh_m3": float(self._model.SEC_membrane),
            "SEC_regen_kWh_m3": float(self._model.SEC_regen),
            "SEC_total_kWh_m3": float(self._model.sec_kWh_m3(incl)),
            "permeate_flow_m3_h": self._model.permeate_flow(cf),
            "concentrate_flow_m3_h": self._model.concentrate_flow(cf),
            "W_elec_kWh_h": self._model.electric_power(cf, incl),
        }

    def get_info(self) -> dict:
        u = self.params["unit"]
        return {
            "name": "Forward Osmosis (FO)",
            "ec_id": self.component_id,
            "fidelity": "F1a",
            "description": (
                "Osmotically driven membrane: SEC_membrane=0.5 kWh/m3 + "
                "draw regeneration=3.0 kWh/m3. recovery=0.80, rejection=0.95."
            ),
            "inputs": {
                "capacity_fraction": {"unit": "dimensionless", "range": [0.1, 1.0]},
                "include_regen": {"unit": "boolean", "default": True},
            },
            "outputs": {
                "recovery": {"unit": "dimensionless"},
                "rejection": {"unit": "dimensionless"},
                "SEC_membrane_kWh_m3": {"unit": "kWh/m3"},
                "SEC_regen_kWh_m3": {"unit": "kWh/m3"},
                "SEC_total_kWh_m3": {"unit": "kWh/m3"},
                "permeate_flow_m3_h": {"unit": "m3/h"},
                "concentrate_flow_m3_h": {"unit": "m3/h"},
                "W_elec_kWh_h": {"unit": "kWh/h"},
            },
            "params": {
                "SEC_membrane": f"{u['SEC_membrane_kWh_m3']['value']} kWh/m3",
                "SEC_regen": f"{u['SEC_regen_kWh_m3']['value']} kWh/m3",
                "recovery": str(u["recovery"]["value"]),
                "draw_agent": u["draw_agent"]["value"],
            },
            "source": "Lutchmiah et al. (2014); Zhao et al. (2012)",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    for regen in [True, False]:
        r = model.predict({"capacity_fraction": 1.0, "include_regen": regen})
        print(f"include_regen={regen}: SEC={r['SEC_total_kWh_m3']:.1f} kWh/m3  "
              f"perm={float(r['permeate_flow_m3_h']):.1f} m3/h  "
              f"W={float(r['W_elec_kWh_h']):.1f} kWh/h")
