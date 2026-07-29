"""
EC084 -- Aquifer Thermal Energy Storage (ATES) -- F2a Doublet-Well Physics-Lumped
Standardised predict() / get_info() interface.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import ATES_F2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for ATES F2a doublet-well physics-lumped model."""

    component_id = "EC084"
    component_name = "Aquifer Thermal Energy Storage (ATES)"
    fidelity = "F2a -- Doublet-Well Physics-Lumped (thermal-radius energy balance)"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            self._raw["unit"].update(params)
        self._model = ATES_F2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        Run seasonal charge/discharge simulation.

        inputs:
            n_cycles    : int   (number of seasonal cycles, default 3)
            T_inj_warm  : float (warm-well injection temp degC, default param)
            V_season    : float (seasonal injected volume m3, default param)
            season_days : float (half-cycle length in days, default param)
        """
        n_cycles = int(inputs.get("n_cycles", 3))
        T_inj = inputs.get("T_inj_warm", None)
        V_season = inputs.get("V_season", None)
        season_days = inputs.get("season_days", None)

        result = self._model.simulate(
            n_cycles=n_cycles, T_inj=T_inj,
            V_season=V_season, season_days=season_days,
        )
        return result

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "n_cycles": {"unit": "-", "range": [1, 30]},
                "T_inj_warm": {"unit": "degC", "range": [12, 90]},
                "V_season": {"unit": "m3", "range": [1000, 1000000]},
                "season_days": {"unit": "day", "range": [30, 365]},
            },
            "outputs": {
                "t": "s",
                "t_days": "day",
                "T_storage": "degC",
                "E_stored_J": "J",
                "E_stored_kWh": "kWh",
                "mode": "+1 charge / -1 discharge",
                "recovery_efficiency": "-",
                "seasonal_efficiency": "- per cycle",
                "thermal_radius_m": "m",
            },
            "source": self._raw.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    r = m.predict({"n_cycles": 3})
    print(f"Thermal radius: {r['thermal_radius_m']:.2f} m")
    print(f"Recovery efficiency (cycle-by-cycle): "
          f"{[round(e, 3) for e in r['seasonal_efficiency']]}")
    print(f"Overall recovery efficiency: {r['recovery_efficiency']:.4f}")
    print(f"E injected: {r['E_injected_kWh']:.1f} kWh, "
          f"E extracted: {r['E_extracted_kWh']:.1f} kWh")
