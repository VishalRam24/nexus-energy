"""EC210 — Electrodialysis — F1a — Standardized Predict Interface"""
import json, os, sys
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from model import ElectrodialysisF1a


class ComponentModel:
    component_id = "EC210"

    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = ElectrodialysisF1a(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        Inputs:
            capacity_fraction    : float or array [0.1, 1.0]
            feed_salinity_ppm    : float [500, 10000], default 4000
        Returns:
            recovery, rejection, SEC_kWh_m3, permeate_flow_m3_h,
            concentrate_flow_m3_h, W_elec_kWh_h
        """
        cf = np.asarray(inputs.get("capacity_fraction", 1.0), dtype=float)
        sal = inputs.get("feed_salinity_ppm", 4000.0)

        return {
            "recovery": float(self._model.recovery),
            "rejection": float(self._model.rejection),
            "SEC_kWh_m3": float(self._model.sec_kWh_m3(sal)) if np.ndim(np.asarray(sal)) == 0 else self._model.sec_kWh_m3(sal),
            "permeate_flow_m3_h": self._model.permeate_flow(cf),
            "concentrate_flow_m3_h": self._model.concentrate_flow(cf),
            "W_elec_kWh_h": self._model.electric_power(cf, sal),
        }

    def get_info(self) -> dict:
        u = self.params["unit"]
        return {
            "name": "Electrodialysis (ED)",
            "ec_id": self.component_id,
            "fidelity": "F1a",
            "description": (
                "Brackish water ED: SEC=1.0 kWh/m3 at 4000 ppm TDS. "
                "recovery=0.85, salt rejection=0.95. Voltage per cell pair: 0.5-1 V."
            ),
            "inputs": {
                "capacity_fraction": {"unit": "dimensionless", "range": [0.1, 1.0]},
                "feed_salinity_ppm": {"unit": "ppm", "range": [500, 10000], "default": 4000},
            },
            "outputs": {
                "recovery": {"unit": "dimensionless"},
                "rejection": {"unit": "dimensionless"},
                "SEC_kWh_m3": {"unit": "kWh/m3"},
                "permeate_flow_m3_h": {"unit": "m3/h"},
                "concentrate_flow_m3_h": {"unit": "m3/h"},
                "W_elec_kWh_h": {"unit": "kWh/h"},
            },
            "params": {
                "SEC_ref": f"{u['SEC_kWh_m3']['value']} kWh/m3 at 4000 ppm",
                "recovery": str(u["recovery"]["value"]),
                "rejection": str(u["rejection"]["value"]),
                "V_cell_pair": f"{u['V_cell_pair_V']['value']} V",
            },
            "source": "Strathmann (2004); GWI DesalData",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    for sal in [1000, 3000, 5000, 8000]:
        r = model.predict({"capacity_fraction": 1.0, "feed_salinity_ppm": sal})
        print(f"Sal={sal} ppm: SEC={r['SEC_kWh_m3']:.3f} kWh/m3  "
              f"perm={float(r['permeate_flow_m3_h']):.1f} m3/h  "
              f"W={float(r['W_elec_kWh_h']):.1f} kWh/h")
