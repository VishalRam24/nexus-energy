"""EC131 — Tidal Barrage — F1b Tidal Range — Standardized Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import TidalBarrageF1b


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = TidalBarrageF1b(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        inputs:
            tidal_range_m   : float or array [m] (peak-to-trough range)
            T_C             : water temperature [°C] (optional)
            S_psu           : salinity [psu] (optional)
            pumping_mode    : bool (optional, default False)
            basin_area_m2   : override [m2] (optional)
        returns:
            power_kw, power_mw, turbine_efficiency, seawater_density_kg_m3,
            theoretical_power_kw, energy_per_cycle_mwh
        """
        R = np.asarray(inputs["tidal_range_m"], dtype=float)
        T_C = inputs.get("T_C", None)
        S_psu = inputs.get("S_psu", None)
        pumping = inputs.get("pumping_mode", False)
        area = inputs.get("basin_area_m2", None)

        m = self._model
        h = R / 2.0

        return {
            "power_kw": m.avg_power_kw(R, area, T_C, S_psu, pumping),
            "power_mw": m.avg_power_mw(R, area, T_C, S_psu, pumping),
            "turbine_efficiency": m.turbine_efficiency(h),
            "seawater_density_kg_m3": m.seawater_density(T_C, S_psu),
            "theoretical_power_kw": m.theoretical_avg_power_w(R, area, T_C, S_psu) / 1000.0,
            "energy_per_cycle_mwh": m.energy_per_cycle_mwh(R, area, T_C, S_psu),
        }

    def get_info(self) -> dict:
        return {
            "name": "Tidal Barrage (Tidal Range)",
            "ec_id": "EC131",
            "fidelity": "F1b",
            "description": (
                "Efficiency vs head amplitude; seawater density(T, S); "
                "sluicing pumping gain option."
            ),
            "inputs": {
                "tidal_range_m": {"unit": "m", "range": [2, 14]},
                "T_C": {"unit": "degC", "range": [5, 20], "optional": True},
                "S_psu": {"unit": "psu", "range": [25, 40], "optional": True},
                "pumping_mode": {"type": "bool", "default": False},
            },
            "outputs": {
                "power_kw": {"unit": "kW"},
                "turbine_efficiency": {"unit": "dimensionless"},
                "seawater_density_kg_m3": {"unit": "kg/m3"},
            },
            "source": "Prandle (1984); Baker (1991); Aggidis & Feather (2012)",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    print("=== EC131 Tidal Barrage F1b ===\n")
    for R in [3.0, 5.0, 8.0, 10.0, 12.0]:
        r = model.predict({"tidal_range_m": R})
        print(f"  R={R:.0f}m  P={float(r['power_mw']):.1f} MW  eta={float(r['turbine_efficiency']):.3f}")
