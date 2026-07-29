"""
EC132 -- Tidal Stream Turbine -- F2a Physics-Lumped Rotor Dynamics
Standardised predict() / get_info() interface.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import TidalStreamTurbineF2a, BETZ_LIMIT

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for the EC132 F2a rotor-dynamics tidal turbine model."""

    component_id = "EC132"
    component_name = "Tidal Stream Turbine"
    fidelity = "F2a -- Physics-Lumped Rotor Dynamics (Cp-lambda + J*dw/dt ODE)"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            self._raw["unit"].update(params)
        self._model = TidalStreamTurbineF2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        Run a dynamic rotor-dynamics simulation under sinusoidal tidal forcing.

        inputs:
            v_mean : float          mean current speed [m/s]      (default 2.0)
            v_amp : float           tidal amplitude [m/s]         (default 1.0)
            tidal_period_s : float  tidal period [s]              (default 44712 ~ M2)
            duration_s : float      simulation length [s]         (default 44712)
            dt : float              output interval [s]           (default 60.0)
            omega0 : float          initial rotor speed [rad/s]   (default optimal)
            water_density : float   seawater density [kg/m3]      (default design)
        """
        v_mean = inputs.get("v_mean", 2.0)
        v_amp = inputs.get("v_amp", 1.0)
        T_tide = inputs.get("tidal_period_s", 44712.0)
        dur = inputs.get("duration_s", 44712.0)
        dt = inputs.get("dt", 60.0)
        omega0 = inputs.get("omega0", None)
        rho = inputs.get("water_density", None)

        return self._model.simulate(
            v_mean=v_mean, v_amp=v_amp, tidal_period_s=T_tide,
            duration_s=dur, dt=dt, omega0=omega0, rho=rho,
        )

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "betz_limit": BETZ_LIMIT,
            "inputs": {
                "v_mean": {"unit": "m/s", "range": [0.0, 4.5]},
                "v_amp": {"unit": "m/s", "range": [0.0, 3.0]},
                "tidal_period_s": {"unit": "s", "range": [3600.0, 90000.0]},
                "duration_s": {"unit": "s", "range": [1.0, 90000.0]},
                "dt": {"unit": "s", "range": [0.5, 600.0]},
                "omega0": {"unit": "rad/s"},
                "water_density": {"unit": "kg/m3", "range": [1000.0, 1035.0]},
            },
            "outputs": {
                "t": "s",
                "v": "m/s",
                "omega": "rad/s",
                "rpm": "rpm",
                "lambda": "-",
                "cp": "-",
                "power_available_w": "W",
                "power_hydro_w": "W",
                "power_electrical_w": "W",
                "power_electrical_kw": "kW",
                "energy_electrical_wh": "Wh",
                "capacity_factor": "-",
            },
            "source": self._raw.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    r = m.predict({"v_mean": 2.0, "v_amp": 1.0, "duration_s": 6000.0, "dt": 60.0})
    print(f"Mean P_elec: {r['power_electrical_kw'].mean():.1f} kW, "
          f"peak Cp: {r['cp'].max():.3f}, "
          f"CF: {r['capacity_factor']:.3f}, "
          f"E: {r['energy_electrical_wh']/1000:.1f} kWh")
