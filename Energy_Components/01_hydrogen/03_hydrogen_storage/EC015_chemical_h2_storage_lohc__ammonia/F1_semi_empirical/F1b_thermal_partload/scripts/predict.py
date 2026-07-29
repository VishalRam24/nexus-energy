"""
EC015 -- Chemical H2 Storage (LOHC / Ammonia) -- F1b Thermal+Part-load -- Standardized Predict Interface
"""

import json
import numpy as np
from pathlib import Path
from model import ChemicalH2StorageF1b


class ComponentModel:
    """Standardized interface for EC015 Chemical H2 Storage -- F1b thermal model."""

    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = ChemicalH2StorageF1b(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        Predict LOHC or ammonia storage energy at temperature and part-load.

        Args:
            inputs: dict with keys:
                - carrier (str):      'lohc' or 'ammonia'
                - direction (str):    'dehydrogenation'/'cracking' or 'hydrogenation'/'synthesis'
                - h2_mass_kg (kg):    Mass of H2 to store/release
                - temperature_K (K):  Reactor temperature (optional, uses nominal if absent)
                - flow_rate_kg_s:     H2 flow rate for part-load (optional)

        Returns:
            dict with thermal_energy_MJ, specific_energy_MJ_per_kg, efficiency,
                      carrier_mass_kg, roundtrip_efficiency
        """
        carrier   = inputs.get("carrier", "lohc")
        direction = inputs.get("direction", "dehydrogenation" if carrier == "lohc" else "cracking")
        m_H2      = np.asarray(inputs.get("h2_mass_kg", 1.0), dtype=float)
        T_K       = inputs.get("temperature_K", None)
        F_kg_s    = inputs.get("flow_rate_kg_s", None)
        if T_K is not None:
            T_K = np.asarray(T_K, dtype=float)
        if F_kg_s is not None:
            F_kg_s = np.asarray(F_kg_s, dtype=float)

        if carrier == "lohc":
            thermal_MJ = self._model.lohc_thermal_energy(m_H2, direction, T_K, F_kg_s)
            spec_MJ    = self._model.lohc_specific_energy(direction, T_K, F_kg_s)
            if direction == "dehydrogenation":
                eta = self._model.lohc_efficiency(T_K if T_K is not None else self._model.lohc_T_d, F_kg_s)
            else:
                eta = np.ones_like(m_H2) * 0.95
            rt_eta     = self._model.lohc_roundtrip_efficiency(T_K, F_kg_s)
            carrier_m  = self._model.lohc_carrier_mass(m_H2)
        else:  # ammonia
            thermal_MJ = self._model.nh3_thermal_energy(m_H2, direction, T_K, F_kg_s)
            spec_MJ    = self._model.nh3_specific_energy(direction, T_K, F_kg_s)
            if direction == "cracking":
                eta = self._model.nh3_efficiency(T_K if T_K is not None else self._model.nh3_T_c, F_kg_s)
            else:
                eta = np.ones_like(m_H2) * 0.95
            rt_eta    = self._model.nh3_roundtrip_efficiency(T_K, F_kg_s)
            carrier_m = self._model.nh3_carrier_mass(m_H2)

        return {
            "thermal_energy_MJ":        thermal_MJ,
            "specific_energy_MJ_per_kg": spec_MJ,
            "efficiency":               eta,
            "carrier_mass_kg":          carrier_m,
            "roundtrip_efficiency":     rt_eta,
        }

    def get_info(self) -> dict:
        return {
            "name": "Chemical H2 Storage (LOHC / Ammonia)",
            "ec_id": "EC015",
            "fidelity": "F1b",
            "description": (
                "Temperature-dependent efficiency (Arrhenius) and part-load correction "
                "for LOHC (DBT) and ammonia H2 storage carriers."
            ),
            "inputs": {
                "carrier":       {"values": ["lohc", "ammonia"]},
                "direction":     {"values": ["dehydrogenation", "hydrogenation", "cracking", "synthesis"]},
                "h2_mass_kg":    {"unit": "kg",            "range": [0.001, 10000.0]},
                "temperature_K": {"unit": "K",             "range": [300.0, 900.0], "optional": True},
                "flow_rate_kg_s": {"unit": "kg/s",         "range": [1e-5, 100.0], "optional": True},
            },
            "outputs": {
                "thermal_energy_MJ":         {"unit": "MJ"},
                "specific_energy_MJ_per_kg": {"unit": "MJ/kg_H2"},
                "efficiency":                {"unit": "dimensionless"},
                "carrier_mass_kg":           {"unit": "kg"},
                "roundtrip_efficiency":      {"unit": "dimensionless"},
            },
            "source": "Preuster (2017); Niermann (2021); Lamb (2019); Reuse (2004)",
        }


if __name__ == "__main__":
    model = ComponentModel()
    print("EC015 Chemical H2 Storage -- F1b Thermal+Part-load")
    print("\n-- LOHC dehydrogenation at different temperatures --")
    for T in [523.15, 573.15, 623.15, 673.15]:
        r = model.predict({"carrier": "lohc", "direction": "dehydrogenation",
                           "h2_mass_kg": 1.0, "temperature_K": T})
        print(f"  T={T:.0f} K: Q={float(r['thermal_energy_MJ']):.3f} MJ/kg, "
              f"eta={float(r['efficiency']):.3f}, rt_eta={float(r['roundtrip_efficiency']):.3f}")
    print("\n-- NH3 cracking at different temperatures --")
    for T in [673.15, 723.15, 773.15, 823.15]:
        r = model.predict({"carrier": "ammonia", "direction": "cracking",
                           "h2_mass_kg": 1.0, "temperature_K": T})
        print(f"  T={T:.0f} K: Q={float(r['thermal_energy_MJ']):.3f} MJ/kg, "
              f"eta={float(r['efficiency']):.3f}")
