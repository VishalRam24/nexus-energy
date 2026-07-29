"""
EC134 -- Oscillating Water Column (OWC) -- F2a Physics-Lumped Dynamics
Standardised predict() / get_info() interface.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import OWC_F2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for OWC F2a physics-lumped dynamics model."""

    component_id = "EC134"
    component_name = "Oscillating Water Column (OWC)"
    fidelity = "F2a -- Physics-Lumped Water-Column + Air-Chamber Dynamics (Wells turbine)"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            self._raw["unit"].update(params)
        self._model = OWC_F2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        Run the coupled OWC dynamic simulation for a regular or irregular wave.

        inputs:
            H_s : float        significant / regular wave height [m]   (default 2.0)
            T_e : float        energy period [s]                       (default 9.0)
            dt : float         time step [s]                           (default 0.05)
            duration_s : float simulation duration [s]                 (default 120.0)
            sea_state : str    "regular" (default) or "irregular" (PM spectrum)
        """
        H_s = inputs.get("H_s", 2.0)
        T_e = inputs.get("T_e", 9.0)
        dt = inputs.get("dt", 0.05)
        dur = inputs.get("duration_s", 120.0)
        sea = inputs.get("sea_state", "regular")

        if sea == "irregular":
            spec = self._model.mean_power_spectrum(H_s, T_e, duration_s=min(dur, 80.0), dt=max(dt, 0.1))
            # also run a representative regular sim for time-series context
            r = self._model.simulate(H_s, T_e, dt=dt, duration_s=dur)
            r["mean_P_elec_W"] = spec["mean_P_elec_W"]
            r["mean_P_elec_kW"] = spec["mean_P_elec_kW"]
            r["mean_P_avail_W"] = spec["mean_P_avail_W"]
            r["capture_width_ratio"] = spec["capture_width_ratio"]
            r["sea_state"] = "irregular_PM"
            return r

        r = self._model.simulate(H_s, T_e, dt=dt, duration_s=dur)
        r["sea_state"] = "regular"
        return r

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "H_s": {"unit": "m", "range": [0.25, 8.0]},
                "T_e": {"unit": "s", "range": [5.0, 20.0]},
                "dt": {"unit": "s", "range": [0.01, 0.5]},
                "duration_s": {"unit": "s", "range": [10.0, 1200.0]},
                "sea_state": {"unit": "-", "options": ["regular", "irregular"]},
            },
            "outputs": {
                "t": "s",
                "x": "m (water-column displacement)",
                "xdot": "m/s (water-column velocity)",
                "pressure": "Pa (chamber gauge pressure)",
                "P_exc": "W (wave excitation power)",
                "P_avail": "W (pneumatic power at turbine)",
                "P_elec": "W (electrical power)",
                "mean_P_elec_kW": "kW",
                "capture_width_ratio": "-",
                "capture_efficiency": "-",
            },
            "source": self._raw.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    r = m.predict({"H_s": 2.0, "T_e": 9.0, "duration_s": 80.0, "dt": 0.05})
    print(f"Mean P_elec: {r['mean_P_elec_kW']:.2f} kW | "
          f"CWR: {r['capture_width_ratio']:.3f} | "
          f"capture_eff: {r['capture_efficiency']:.3f} | "
          f"omega/omega_n: {r['omega']/r['omega_n']:.3f}")
