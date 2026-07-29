"""EC208 — CO2 Geological Sequestration — F1a Injection Model — Standardized Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import CO2SequestrationF1a


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = CO2SequestrationF1a(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        Predict CO2 geological sequestration injection and storage.

        Args:
            inputs: dict with keys:
                - P_wellhead_bar      : wellhead injection pressure [bar], default 150
                - depth_m             : reservoir depth [m], default 2000
                - k_mD                : reservoir permeability [mD], default 50
                - h_m                 : reservoir thickness [m], default 100
                - area_km2            : reservoir area [km²], default 100
                - porosity            : reservoir porosity [-], default 0.15
                - storage_efficiency  : CO2 storage efficiency factor [-], default 0.02

        Returns:
            dict with keys:
                - bottomhole_P_bar        : bottomhole pressure [bar]
                - injection_rate_kg_per_s : CO2 injection rate [kg/s]
                - injection_rate_tco2_per_day : injection rate [tCO2/day]
                - storage_capacity_tco2   : total effective storage [tCO2]
                - pore_volume_m3          : total pore volume [m³]
                - years_to_fill           : years at this injection rate to fill reservoir
        """
        P_wh = np.asarray(inputs.get("P_wellhead_bar", 150.0), dtype=float)
        depth = np.asarray(inputs.get("depth_m", 2000.0), dtype=float)
        k = np.asarray(inputs.get("k_mD", 50.0), dtype=float)
        h = np.asarray(inputs.get("h_m", 100.0), dtype=float)
        area = np.asarray(inputs.get("area_km2", 100.0), dtype=float)
        phi = np.asarray(inputs.get("porosity", 0.15), dtype=float)
        eff = np.asarray(inputs.get("storage_efficiency", 0.02), dtype=float)

        P_bh = self._model.bottomhole_pressure_pa(P_wh, depth)
        m_dot = self._model.injection_rate_kg_per_s(P_wh, depth, k, h)
        capacity = self._model.storage_capacity_tco2(area, h, phi, eff)
        pore_vol = self._model.storage_capacity_pore_volume_m3(area, h, phi)
        years = self._model.years_to_fill(m_dot, area, h, phi, eff)

        return {
            "bottomhole_P_bar":           P_bh / 1e5,
            "injection_rate_kg_per_s":    m_dot,
            "injection_rate_tco2_per_day": self._model.injection_rate_tco2_per_day(P_wh, depth, k, h),
            "storage_capacity_tco2":      capacity,
            "pore_volume_m3":             pore_vol,
            "years_to_fill":              years,
        }

    def get_info(self) -> dict:
        return {
            "name": "CO2 Geological Sequestration",
            "ec_id": "EC208",
            "fidelity": "F1a",
            "description": "Darcy radial injection model + pore-volume storage capacity",
            "inputs": {
                "P_wellhead_bar":    {"unit": "bar", "range": [80.0, 300.0], "default": 150.0},
                "depth_m":           {"unit": "m",   "range": [800.0, 5000.0], "default": 2000.0},
                "k_mD":              {"unit": "mD",  "range": [0.1, 5000.0], "default": 50.0},
                "h_m":               {"unit": "m",   "range": [10.0, 500.0], "default": 100.0},
                "area_km2":          {"unit": "km²", "range": [1.0, 10000.0], "default": 100.0},
                "porosity":          {"unit": "-",   "range": [0.05, 0.35], "default": 0.15},
                "storage_efficiency":{"unit": "-",   "range": [0.005, 0.10], "default": 0.02},
            },
            "outputs": {
                "bottomhole_P_bar":           {"unit": "bar"},
                "injection_rate_kg_per_s":    {"unit": "kg/s"},
                "injection_rate_tco2_per_day":{"unit": "tCO2/day"},
                "storage_capacity_tco2":      {"unit": "tCO2"},
                "pore_volume_m3":             {"unit": "m³"},
                "years_to_fill":              {"unit": "years"},
            },
            "source": "van der Meer (1993); IPCC (2005) CCS SR Ch.5; Benson & Cole (2008)",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    print("Default reservoir (2000m, 50mD, 100km², E=2%):")
    r = model.predict({})
    for k, v in r.items():
        print(f"  {k}: {float(v):.4g}")

    print("\nSensitivity to wellhead pressure (100-250 bar):")
    for P_wh in [100, 125, 150, 200, 250]:
        r = model.predict({"P_wellhead_bar": float(P_wh)})
        print(f"  P_wh={P_wh} bar: m_dot={float(r['injection_rate_kg_per_s']):.2f} kg/s, "
              f"tCO2/day={float(r['injection_rate_tco2_per_day']):.0f}")
