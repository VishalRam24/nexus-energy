"""
EC066 -- Offshore Floating Wind Turbine -- F2a BEM + Floating Aero-Hydro
Standardised predict() / get_info() interface (mirrors EC001 F2a template).
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import FloatingWindF2a, BETZ

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for the EC066 F2a aero-hydro coupled lumped model."""

    component_id = "EC066"
    component_name = "Offshore Floating Wind Turbine"
    fidelity = "F2a -- BEM Rotor + Floating-Platform Surge/Pitch Aero-Hydro ODE"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            self._raw["unit"].update(params)
        self._model = FloatingWindF2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        Run the coupled aero-hydro time-domain simulation.

        inputs:
            wind_speed_ms : float (or list/callable) free hub-height wind speed
            blade_pitch_deg : float  (default 0.0)
            dt : float            (default 0.05 s)
            duration_s : float    (default 120.0 s)
            wave_height_m : float (default 0.0 -> calm)
            wave_period_s : float (default 10.0 s)

        returns dict of time-series arrays plus scalar summaries.
        """
        ws = inputs.get("wind_speed_ms", 10.59)
        beta = inputs.get("blade_pitch_deg", 0.0)
        dt = inputs.get("dt", 0.05)
        dur = inputs.get("duration_s", 120.0)
        Hw = inputs.get("wave_height_m", 0.0)
        Tw = inputs.get("wave_period_s", 10.0)

        # allow lists to be treated as a step/interpolated profile
        if isinstance(ws, (list, tuple)):
            import numpy as np
            arr = np.asarray(ws, dtype=float)
            tgrid = np.linspace(0.0, dur, len(arr))
            ws = lambda t, _a=arr, _g=tgrid: float(np.interp(t, _g, _a))

        r = self._model.simulate(ws, beta_deg=beta, dt=dt, duration_s=dur,
                                 H_wave=Hw, T_wave=Tw)

        # scalar summaries
        r["power_elec_mean_MW"] = float(r["power_elec"].mean() / 1e6)
        r["cp_max"] = float(r["cp"].max())
        r["surge_peak_m"] = float(abs(r["surge"]).max())
        r["pitch_peak_deg"] = float(abs(r["pitch_deg"]).max())
        return r

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "betz_limit": BETZ,
            "inputs": {
                "wind_speed_ms": {"unit": "m/s", "range": [0, 30]},
                "blade_pitch_deg": {"unit": "deg", "range": [-2, 30]},
                "dt": {"unit": "s", "range": [0.001, 1.0]},
                "duration_s": {"unit": "s", "range": [1, 1200]},
                "wave_height_m": {"unit": "m", "range": [0, 10]},
                "wave_period_s": {"unit": "s", "range": [4, 20]},
            },
            "outputs": {
                "t": "s",
                "rotor_speed": "rad/s",
                "cp": "-",
                "V_rel": "m/s",
                "power_aero": "W",
                "power_elec": "W",
                "thrust": "N",
                "surge": "m",
                "pitch_deg": "deg",
            },
            "source": self._raw.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    info = m.get_info()
    print(f"{info['component_id']} {info['component_name']} -- {info['fidelity']}")
    print(f"Betz limit = {info['betz_limit']:.4f}")
    # below-rated, with moderate sea state
    r = m.predict({"wind_speed_ms": 9.0, "duration_s": 60.0, "dt": 0.1,
                   "wave_height_m": 3.0, "wave_period_s": 10.0})
    print(f"Mean elec power : {r['power_elec_mean_MW']:.2f} MW")
    print(f"Final rotor speed: {r['rotor_speed'][-1]:.4f} rad/s "
          f"({r['rotor_speed'][-1]*60/(2*3.14159):.2f} rpm)")
    print(f"Peak Cp          : {r['cp_max']:.4f} (Betz {BETZ:.4f})")
    print(f"Peak surge       : {r['surge_peak_m']:.3f} m")
    print(f"Peak pitch       : {r['pitch_peak_deg']:.3f} deg")
