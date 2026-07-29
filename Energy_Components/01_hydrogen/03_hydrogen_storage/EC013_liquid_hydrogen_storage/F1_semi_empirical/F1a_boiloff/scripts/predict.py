"""EC013 — Liquid H2 Storage — F1a Boil-off — Standardized Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import LH2F1a


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = LH2F1a(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        Predict LH2 storage state and boil-off.

        Args:
            inputs: dict with keys:
                - fill_fraction (-):    scalar or array, 0-0.95
                - T_ambient     (K):    optional, default 298.15

        Returns:
            dict with keys:
                - stored_mass_kg, energy_stored_MJ, heat_leak_W,
                  boiloff_kg_per_day, boiloff_pct_per_day,
                  time_to_empty_days, gravimetric_wt_pct, volumetric_kg_per_m3
        """
        f = np.asarray(inputs["fill_fraction"], dtype=float)
        T_amb = np.asarray(inputs.get("T_ambient", 298.15), dtype=float)

        m_dot = self._model.boiloff_mass_rate(T_amb)

        return {
            "stored_mass_kg":      self._model.stored_mass(f),
            "energy_stored_MJ":    self._model.energy_stored(f),
            "heat_leak_W":         self._model.heat_leak(T_amb),
            "boiloff_kg_per_day":  m_dot * 86400.0,
            "boiloff_pct_per_day": self._model.boiloff_rate_percent_day(f, T_amb),
            "time_to_empty_days":  self._model.time_to_empty_days(f, T_amb),
            "gravimetric_wt_pct":  self._model.gravimetric_density(f),
            "volumetric_kg_per_m3": self._model.volumetric_density(f),
        }

    def get_info(self) -> dict:
        return {
            "name": "Liquid Hydrogen (LH2) Storage",
            "ec_id": "EC013",
            "fidelity": "F1a",
            "description": "Heat-leak driven boil-off: Q=UA(T_amb-T_sat); m_dot_BO = Q/h_vap",
            "inputs": {
                "fill_fraction": {"unit": "dimensionless", "range": [0.0, 0.95]},
                "T_ambient":     {"unit": "K",            "range": [233.0, 333.0], "default": 298.15},
            },
            "outputs": {
                "stored_mass_kg":       {"unit": "kg"},
                "energy_stored_MJ":     {"unit": "MJ"},
                "heat_leak_W":          {"unit": "W"},
                "boiloff_kg_per_day":   {"unit": "kg/day"},
                "boiloff_pct_per_day":  {"unit": "%/day"},
                "time_to_empty_days":   {"unit": "days"},
                "gravimetric_wt_pct":   {"unit": "wt%"},
                "volumetric_kg_per_m3": {"unit": "kg/m3"},
            },
            "source": "Sherif et al. (1997); Petitpas (2018) NREL; Notardonato et al. (2017)",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    for f in [0.25, 0.5, 0.75, 0.95]:
        r = model.predict({"fill_fraction": f, "T_ambient": 298.15})
        print(f"fill={f:.2f}: m={float(r['stored_mass_kg']):.2f} kg, "
              f"BOR={float(r['boiloff_pct_per_day']):.2f} %/day, "
              f"t_empty={float(r['time_to_empty_days']):.0f} days")
