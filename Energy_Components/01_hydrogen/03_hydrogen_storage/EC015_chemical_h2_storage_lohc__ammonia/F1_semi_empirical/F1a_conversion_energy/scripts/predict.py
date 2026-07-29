"""EC015 — Chemical H2 Storage (LOHC/Ammonia) — F1a Conversion Energy — Standardized Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import ChemicalH2StorageF1a


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = ChemicalH2StorageF1a(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        Predict chemical H2 storage conversion energy and carrier requirements.

        Args:
            inputs: dict with keys:
                - h2_mass_kg (kg):     Mass of H2 to store or release
                - mode (str):          'lohc' or 'ammonia'
                - direction (str):     For LOHC: 'hydrogenation' or 'dehydrogenation'
                                       For NH3:  'synthesis' or 'cracking'

        Returns:
            dict with keys:
                - carrier_mass_kg, thermal_energy_MJ, specific_energy_MJ_per_kg,
                  reactor_temperature_K, roundtrip_efficiency, gravimetric_capacity_wt_pct
        """
        m_H2 = np.asarray(inputs["h2_mass_kg"], dtype=float)
        mode = inputs.get("mode", "lohc").lower()
        direction = inputs.get("direction", "dehydrogenation" if mode == "lohc" else "cracking")

        if mode == "lohc":
            carrier = self._model.lohc_carrier_mass(m_H2)
            Q = self._model.lohc_thermal_energy(m_H2, direction=direction)
            q_spec = self._model.lohc_specific_energy(direction=direction)
            T_rxn = self._model.lohc_reactor_temperature(direction=direction)
            eta_rt = self._model.lohc_roundtrip_efficiency()
            cap_wt = self._model.lohc_cap_wt
        else:  # ammonia
            carrier = self._model.nh3_carrier_mass(m_H2)
            Q = self._model.nh3_thermal_energy(m_H2, direction=direction)
            q_spec = self._model.nh3_specific_energy(direction=direction)
            T_rxn = self._model.nh3_reactor_temperature(direction=direction)
            eta_rt = self._model.nh3_roundtrip_efficiency()
            cap_wt = self._model.nh3_cap_wt

        return {
            "carrier_mass_kg":              carrier,
            "thermal_energy_MJ":            Q,
            "specific_energy_MJ_per_kg_H2": float(q_spec),
            "reactor_temperature_K":        float(T_rxn),
            "roundtrip_efficiency":         float(eta_rt),
            "gravimetric_capacity_wt_pct":  float(cap_wt),
        }

    def get_info(self) -> dict:
        return {
            "name": "Chemical H2 Storage (LOHC / Ammonia)",
            "ec_id": "EC015",
            "fidelity": "F1a",
            "description": (
                "Hydrogenation/dehydrogenation energy balance for LOHC (DBT, 6.2 wt%) "
                "and Ammonia (17.6 wt%). ΔH per mol H2 from literature."
            ),
            "inputs": {
                "h2_mass_kg": {"unit": "kg",  "range": [0.001, 10000.0]},
                "mode":       {"unit": "str", "values": ["lohc", "ammonia"]},
                "direction":  {"unit": "str", "values": ["hydrogenation", "dehydrogenation",
                                                          "synthesis", "cracking"]},
            },
            "outputs": {
                "carrier_mass_kg":              {"unit": "kg"},
                "thermal_energy_MJ":            {"unit": "MJ", "note": "+ve = heat required, -ve = heat released"},
                "specific_energy_MJ_per_kg_H2": {"unit": "MJ/kg_H2"},
                "reactor_temperature_K":        {"unit": "K"},
                "roundtrip_efficiency":         {"unit": "dimensionless"},
                "gravimetric_capacity_wt_pct":  {"unit": "wt%"},
            },
            "source": "Preuster et al. (2017) Acc. Chem. Res.; Niermann et al. (2021) Energy Environ. Sci.; Lamb et al. (2019)",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    print("EC015 Chemical H2 Storage — F1a Conversion Energy")
    for m in [1, 10, 100]:
        r_lohc = model.predict({"h2_mass_kg": m, "mode": "lohc", "direction": "dehydrogenation"})
        r_nh3 = model.predict({"h2_mass_kg": m, "mode": "ammonia", "direction": "cracking"})
        print(f"  H2={m} kg:")
        print(f"    LOHC: carrier={float(r_lohc['carrier_mass_kg']):.1f} kg, "
              f"Q={float(r_lohc['thermal_energy_MJ']):.2f} MJ, "
              f"T={r_lohc['reactor_temperature_K']:.0f} K")
        print(f"    NH3:  carrier={float(r_nh3['carrier_mass_kg']):.2f} kg, "
              f"Q={float(r_nh3['thermal_energy_MJ']):.2f} MJ, "
              f"T={r_nh3['reactor_temperature_K']:.0f} K")
