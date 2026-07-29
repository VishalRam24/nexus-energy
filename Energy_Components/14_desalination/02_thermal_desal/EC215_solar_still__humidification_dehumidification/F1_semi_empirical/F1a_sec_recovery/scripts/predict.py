"""EC215 — Solar Still / HDH — F1a — Standardized Predict Interface"""
import json, os, sys
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from model import SolarStillHDHF1a


class ComponentModel:
    component_id = "EC215"

    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = SolarStillHDHF1a(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        Inputs:
            capacity_fraction       : float or array [0.1, 1.0]
            solar_irradiance_W_m2   : float [400, 1200], default 800
        Returns:
            mode, GOR, yield_L_h, yield_m3_h, solar_power_W, SEC_solar_kWh_m3
        """
        cf = np.asarray(inputs.get("capacity_fraction", 1.0), dtype=float)
        G = inputs.get("solar_irradiance_W_m2", None)

        return {
            "mode": self._model.mode,
            "GOR": float(self._model.GOR(G)) if (G is None or np.ndim(np.asarray(G)) == 0) else self._model.GOR(G),
            "yield_L_h": self._model.yield_L_h(cf, G),
            "yield_m3_h": self._model.yield_m3_h(cf, G),
            "solar_power_W": self._model.solar_power(cf, G),
            "SEC_solar_kWh_m3": float(self._model.sec_solar_kWh_m3(G)),
        }

    def get_info(self) -> dict:
        u = self.params["unit"]
        return {
            "name": "Solar Still / Humidification-Dehumidification (HDH)",
            "ec_id": self.component_id,
            "fidelity": "F1a",
            "description": (
                f"Mode: {u['mode']['value']}. HDH GOR=1-3, solar still productivity=4-6 L/(m2*day). "
                "Passive solar-driven desalination. SEC_solar~600 kWh/m3."
            ),
            "inputs": {
                "capacity_fraction": {"unit": "dimensionless", "range": [0.1, 1.0]},
                "solar_irradiance_W_m2": {"unit": "W/m2", "range": [400, 1200], "default": 800},
            },
            "outputs": {
                "mode": {"unit": "string"},
                "GOR": {"unit": "dimensionless"},
                "yield_L_h": {"unit": "L/h"},
                "yield_m3_h": {"unit": "m3/h"},
                "solar_power_W": {"unit": "W"},
                "SEC_solar_kWh_m3": {"unit": "kWh/m3"},
            },
            "params": {
                "mode": u["mode"]["value"],
                "GOR_HDH": f"{u['GOR_HDH']['value']} (range 1-3)",
                "productivity": f"{u['productivity_L_m2_day']['value']} L/(m2*day)",
                "collector_area": f"{u['collector_area_m2']['value']} m2",
                "G_ref": f"{u['solar_irradiance_W_m2']['value']} W/m2",
            },
            "source": "Kaushal & Varun (2010); Narayan et al. (2012)",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    for G in [400, 600, 800, 1000, 1200]:
        r = model.predict({"capacity_fraction": 1.0, "solar_irradiance_W_m2": G})
        print(f"G={G:4d} W/m2: GOR={r['GOR']:.2f}  yield={float(r['yield_L_h']):.3f} L/h  "
              f"yield={float(r['yield_m3_h'])*1000:.2f} L/h")
