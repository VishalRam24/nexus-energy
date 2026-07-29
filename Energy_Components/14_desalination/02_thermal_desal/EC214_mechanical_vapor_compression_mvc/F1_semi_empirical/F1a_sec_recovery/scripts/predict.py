"""EC214 — MVC — F1a — Standardized Predict Interface"""
import json, os, sys
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from model import MVCF1a


class ComponentModel:
    component_id = "EC214"

    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = MVCF1a(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        Inputs:
            capacity_fraction  : float or array [0.1, 1.0]
            compression_ratio  : float [1.05, 1.5], default 1.2
        Returns:
            recovery, SEC_kWh_m3, distillate_flow_m3_h,
            concentrate_flow_m3_h, W_elec_kWh_h, GOR_equiv
        """
        cf = np.asarray(inputs.get("capacity_fraction", 1.0), dtype=float)
        CR = inputs.get("compression_ratio", None)

        return {
            "recovery": float(self._model.recovery),
            "SEC_kWh_m3": float(self._model.sec_kWh_m3(CR)) if (CR is None or np.ndim(np.asarray(CR)) == 0) else self._model.sec_kWh_m3(CR),
            "distillate_flow_m3_h": self._model.distillate_flow(cf),
            "concentrate_flow_m3_h": self._model.concentrate_flow(cf),
            "W_elec_kWh_h": self._model.electric_power(cf, CR),
            "GOR_equiv": float(self._model.GOR_equiv),
        }

    def get_info(self) -> dict:
        u = self.params["unit"]
        return {
            "name": "Mechanical Vapor Compression (MVC)",
            "ec_id": self.component_id,
            "fidelity": "F1a",
            "description": (
                "All-electric desalination: SEC=8-12 kWh/m3, recovery=0.50. "
                "No external thermal input. Compressor drives heat cycle."
            ),
            "inputs": {
                "capacity_fraction": {"unit": "dimensionless", "range": [0.1, 1.0]},
                "compression_ratio": {"unit": "dimensionless", "range": [1.05, 1.5], "default": 1.2},
            },
            "outputs": {
                "recovery": {"unit": "dimensionless"},
                "SEC_kWh_m3": {"unit": "kWh/m3"},
                "distillate_flow_m3_h": {"unit": "m3/h"},
                "concentrate_flow_m3_h": {"unit": "m3/h"},
                "W_elec_kWh_h": {"unit": "kWh/h"},
                "GOR_equiv": {"unit": "dimensionless"},
            },
            "params": {
                "SEC_range": f"{u['SEC_min_kWh_m3']['value']}-{u['SEC_max_kWh_m3']['value']} kWh/m3",
                "recovery": str(u["recovery"]["value"]),
                "compression_ratio": str(u["compression_ratio"]["value"]),
                "compressor_efficiency": str(u["compressor_efficiency"]["value"]),
            },
            "source": "Mistry et al. (2011); GWI DesalData",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    for CR in [1.05, 1.1, 1.2, 1.4]:
        r = model.predict({"capacity_fraction": 1.0, "compression_ratio": CR})
        print(f"CR={CR:.2f}: SEC={r['SEC_kWh_m3']:.2f} kWh/m3  "
              f"dist={float(r['distillate_flow_m3_h']):.1f} m3/h  "
              f"W={float(r['W_elec_kWh_h']):.0f} kWh/h")
