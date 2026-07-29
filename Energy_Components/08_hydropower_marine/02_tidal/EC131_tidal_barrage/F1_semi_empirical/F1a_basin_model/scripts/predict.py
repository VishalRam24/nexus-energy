"""EC131 — Tidal Barrage — F1a Basin Model — Standardized Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import TidalBarrageF1a


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = TidalBarrageF1a(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        inputs:
            tidal_range_m   : full peak-to-trough tidal range [m]
            basin_area_m2   : optional override [m2]
        """
        R = np.asarray(inputs["tidal_range_m"], dtype=float)
        A = inputs.get("basin_area_m2", None)
        return {
            "avg_power_kw": self._model.avg_power_kw(R, A),
            "avg_power_mw": self._model.avg_power_mw(R, A),
            "theoretical_power_kw": self._model.theoretical_avg_power_w(R, A) / 1000.0,
            "energy_per_cycle_mwh": self._model.energy_per_cycle_mwh(R, A),
        }

    def get_info(self) -> dict:
        return {
            "name": "Tidal Barrage",
            "ec_id": "EC131",
            "fidelity": "F1a",
            "description": "Ebb-generation basin model: P_avg = eta * 0.5*rho*g*A*h^2 / T_tide; h = tidal_range/2",
            "inputs": {
                "tidal_range_m": {"unit": "m", "range": [0.0, 16.0], "note": "Full peak-to-trough range"},
                "basin_area_m2": {"unit": "m2", "range": [1e5, 1e9], "note": "Optional, defaults to design value"},
            },
            "outputs": {
                "avg_power_kw": {"unit": "kW"},
                "avg_power_mw": {"unit": "MW"},
                "theoretical_power_kw": {"unit": "kW", "note": "Before plant efficiency"},
                "energy_per_cycle_mwh": {"unit": "MWh"},
            },
            "source": "Prandle (1984); Baker (1991); Charlier (2003)",
        }


if __name__ == "__main__":
    model = ComponentModel()
    r = model.predict({"tidal_range_m": 16.0})   # La Rance design ~8m amplitude = 16m range
    print(f"Design point (R=16m): P_avg={float(r['avg_power_mw']):.1f} MW, "
          f"E_cycle={float(r['energy_per_cycle_mwh']):.1f} MWh")
