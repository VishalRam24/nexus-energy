"""EC209 — Reverse Osmosis (RO) — F1a SEC Model — Standardized Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import ROF1a


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = ROF1a(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        inputs:
            feed_salinity  : float or array — feed salinity [g/L]
            recovery       : float or array — water recovery fraction [0–1]
            feed_flow_m3h  : float or array — feed flow rate [m3/hr]
        returns:
            sec_kwhm3            : specific energy consumption [kWh/m3 permeate]
            permeate_flow_m3h    : permeate (product water) flow [m3/hr]
            feed_pressure_bar    : required feed pressure [bar]
            permeate_salinity_gl : product water salinity [g/L]
        """
        S = np.asarray(inputs["feed_salinity"], dtype=float)
        r = np.asarray(inputs["recovery"], dtype=float)
        Q = np.asarray(inputs.get("feed_flow_m3h", 100.0), dtype=float)
        return self._model.compute(S, r, Q)

    def get_info(self) -> dict:
        return {
            "name": "Reverse Osmosis (RO)",
            "ec_id": "EC209",
            "fidelity": "F1a",
            "description": "SEC = f(feed_salinity, recovery) with energy recovery device (ERD)",
            "inputs": {
                "feed_salinity":  {"unit": "g/L",   "range": [1.0, 45.0]},
                "recovery":       {"unit": "-",     "range": [0.2, 0.6]},
                "feed_flow_m3h":  {"unit": "m3/hr", "range": [10.0, 1000.0], "default": 100.0},
            },
            "outputs": {
                "sec_kwhm3":            {"unit": "kWh/m3"},
                "permeate_flow_m3h":    {"unit": "m3/hr"},
                "feed_pressure_bar":    {"unit": "bar"},
                "permeate_salinity_gl": {"unit": "g/L"},
            },
            "source": "Elimelech & Phillip (2011), Science, 333, 712-717",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    r = model.predict({"feed_salinity": 35.0, "recovery": 0.45, "feed_flow_m3h": 100.0})
    print(f"SWRO (35 g/L, r=0.45): SEC={float(r['sec_kwhm3']):.2f} kWh/m3, "
          f"P_feed={float(r['feed_pressure_bar']):.1f} bar, "
          f"Q_perm={float(r['permeate_flow_m3h']):.1f} m3/hr")
