"""
EC067 -- Airborne Wind Energy (AWE) -- F2a Crosswind Pumping-Cycle
Standardised predict() / get_info() interface (mirrors EC001 F2a).
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import AWE_PumpingCycle_F2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for AWE F2a crosswind pumping-cycle model."""

    component_id = "EC067"
    component_name = "Airborne Wind Energy (AWE)"
    fidelity = "F2a -- Crosswind Pumping-Cycle with Tether-Length ODE"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            self._raw["unit"].update(params)
        self._model = AWE_PumpingCycle_F2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        Simulate one pumping cycle at a given wind speed.

        inputs:
            v_wind : float   wind speed at altitude [m/s]
            n_eval : int     samples per phase (default 200)
        """
        v_wind = inputs.get("v_wind", 10.0)
        n_eval = int(inputs.get("n_eval", 200))
        return self._model.simulate(v_wind, n_eval=n_eval)

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "v_wind": {"unit": "m/s", "range": [0, 50]},
                "n_eval": {"unit": "-", "range": [10, 2000]},
            },
            "outputs": {
                "t": "s (cycle time-series)",
                "L": "m (tether length)",
                "phase": "+1 reel-out / -1 reel-in",
                "P_elec": "W (instantaneous electrical power)",
                "F_traction": "N (tether tension)",
                "P_loyd_limit": "W (Loyd 1980 ideal upper bound)",
                "P_avg": "W (cycle-average net electrical power)",
                "duty": "- (reel-out fraction of cycle)",
                "traction_peak": "N",
                "capacity_factor": "-",
                "E_out": "J", "E_in": "J", "E_net": "J",
                "energy_residual": "- (energy-balance check)",
            },
            "source": self._raw.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    r = m.predict({"v_wind": 10.0})
    print(
        f"v=10 m/s | P_avg={r['P_avg']/1000:.2f} kW "
        f"(Loyd limit {r['P_loyd_limit']/1000:.2f} kW) | "
        f"duty={r['duty']:.2f} | F_peak={r['traction_peak']/1000:.2f} kN | "
        f"t_cycle={r['t_cycle']:.1f} s | CF={r['capacity_factor']:.2f}"
    )
