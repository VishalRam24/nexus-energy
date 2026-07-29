"""
EC215 -- Solar Still / HDH -- F2a Basin Still (Dunkle)
Standardised predict() / get_info() interface.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import SolarStillF2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for EC215 F2a single-basin solar still."""

    component_id = "EC215"
    component_name = "Solar Still / Humidification-Dehumidification (HDH)"
    fidelity = "F2a -- Single-Basin Solar Still, Dunkle Lumped Energy Balance"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            self._raw["unit"].update(params)
        self._model = SolarStillF2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        Simulate one (or multi-) day solar-still operation.

        inputs:
            G_peak_W_m2 : float   peak clear-sky irradiance (default 900)
            T_amb_K     : float   ambient temperature (default 298.15)
            T_water0_K  : float   initial water temp (default = ambient)
            T_glass0_K  : float   initial glass temp (default = ambient)
            water_depth_mm : float  optional -- rescales basin water mass
            duration_s  : float   horizon (default 86400 = 1 day)
            dt          : float   output step (default 600 s)
        """
        Gp = inputs.get("G_peak_W_m2", self._model.G_peak)
        T_amb = inputs.get("T_amb_K", self._model.T_amb)
        T_w0 = inputs.get("T_water0_K", None)
        T_g0 = inputs.get("T_glass0_K", None)
        dur = inputs.get("duration_s", 86400.0)
        dt = inputs.get("dt", 600.0)

        depth = inputs.get("water_depth_mm", None)
        if depth is not None:
            # m_water = rho * A * depth ; rho=1000 kg/m3, depth in mm
            self._model.m_w = 1000.0 * self._model.A * (depth / 1000.0)

        r = self._model.simulate(
            G_peak=Gp, T_w0=T_w0, T_g0=T_g0, T_amb=T_amb,
            duration_s=dur, dt=dt,
        )
        return r

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "G_peak_W_m2": {"unit": "W/m2", "range": [0, 1200]},
                "T_amb_K": {"unit": "K", "range": [273.15, 318.15]},
                "T_water0_K": {"unit": "K", "range": [283.15, 373.15]},
                "T_glass0_K": {"unit": "K", "range": [273.15, 360.15]},
                "water_depth_mm": {"unit": "mm", "range": [5, 100]},
                "duration_s": {"unit": "s"},
                "dt": {"unit": "s"},
            },
            "outputs": {
                "t": "s",
                "T_water": "K",
                "T_glass": "K",
                "G": "W/m2",
                "q_evap": "W/m2",
                "q_conv": "W/m2",
                "q_rad": "W/m2",
                "distillate_rate_L_h": "L/h",
                "cumulative_distillate_kg": "kg",
                "daily_yield_L_m2": "L/(m2.day)",
            },
            "source": self._raw.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    r = m.predict({"G_peak_W_m2": 900.0, "duration_s": 86400.0, "dt": 600.0})
    print(f"Peak T_water: {r['T_water'].max() - 273.15:.1f} C, "
          f"daily yield: {r['daily_yield_L_m2']:.2f} L/(m2.day)")
