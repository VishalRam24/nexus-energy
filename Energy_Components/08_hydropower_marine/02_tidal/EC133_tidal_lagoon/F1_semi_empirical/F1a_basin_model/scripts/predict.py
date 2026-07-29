"""EC133 — Tidal Lagoon — F1a Basin Model — Standardized Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import TidalLagoonF1a


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = TidalLagoonF1a(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        inputs:
            tidal_range_m  : full peak-to-trough tidal range [m]
            lagoon_area_m2 : optional area override [m2]
        """
        R = np.asarray(inputs["tidal_range_m"], dtype=float)
        A = inputs.get("lagoon_area_m2", None)
        return {
            "avg_power_kw": self._model.avg_power_kw(R, A),
            "avg_power_mw": self._model.avg_power_mw(R, A),
            "theoretical_power_kw": self._model.theoretical_avg_power_w(R, A) / 1000.0,
            "energy_per_cycle_mwh": self._model.energy_per_cycle_mwh(R, A),
        }

    def get_info(self) -> dict:
        return {
            "name": "Tidal Lagoon",
            "ec_id": "EC133",
            "fidelity": "F1a",
            "description": "Bidirectional basin model: P_avg = eta * n_cycles * 0.5*rho*g*A*h^2 / T_tide; "
                           "both ebb and flood generation; offshore enclosed lagoon",
            "inputs": {
                "tidal_range_m": {"unit": "m", "range": [0.0, 12.0], "note": "Full peak-to-trough range"},
                "lagoon_area_m2": {"unit": "m2", "range": [1e5, 5e8], "note": "Optional override"},
            },
            "outputs": {
                "avg_power_kw": {"unit": "kW"},
                "avg_power_mw": {"unit": "MW"},
                "theoretical_power_kw": {"unit": "kW", "note": "Before plant efficiency"},
                "energy_per_cycle_mwh": {"unit": "MWh"},
            },
            "source": "Aggidis & Feather (2012); Xiao et al. (2020); Tidal Lagoon Power (2015)",
        }


if __name__ == "__main__":
    model = ComponentModel()
    # Swansea Bay: A=11.5 km2, R~9m tidal range, eta~0.25
    r = model.predict({"tidal_range_m": 9.0})
    print(f"Swansea Bay proxy (R=9m): P_avg={float(r['avg_power_mw']):.1f} MW, "
          f"E_cycle={float(r['energy_per_cycle_mwh']):.1f} MWh")
