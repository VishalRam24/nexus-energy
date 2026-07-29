"""
EC082 -- Ice Thermal Storage -- F2a Stefan-Problem Enthalpy Model
Standardised predict() / get_info() interface (mirrors EC001 template).
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import IceTES_F2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for the EC082 ice TES F2a enthalpy-method model."""

    component_id = "EC082"
    component_name = "Ice Thermal Storage"
    fidelity = "F2a -- Stefan-Problem Lumped Enthalpy Method (moving freeze/melt front)"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            self._raw["unit"].update(params)
        self._model = IceTES_F2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        Run a dynamic charge/discharge simulation.

        inputs:
            T_brine_C     : float or callable(t)->C  coil supply temperature
                            (< 0 charges/builds ice, > 0 discharges/melts)
            T_amb_C       : float (default 20.0)      ambient for shell losses
            ice_fraction0 : float (default 0.0)       initial ice fraction [0,1]
            dt            : float (default 60.0)      output interval [s]
            duration_s    : float (default 3600.0)    total simulated time [s]
        """
        T_brine = inputs.get("T_brine_C", self._raw["unit"]["T_brine_charge"]["value"])
        T_amb = inputs.get("T_amb_C", self._raw["unit"]["T_amb_default"]["value"])
        f0 = inputs.get("ice_fraction0", 0.0)
        dt = inputs.get("dt", 60.0)
        dur = inputs.get("duration_s", 3600.0)

        return self._model.simulate(T_brine, T_amb, f0, dt, dur)

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "T_brine_C": {"unit": "degC", "range": [-12.0, 25.0]},
                "T_amb_C": {"unit": "degC", "range": [-10.0, 40.0]},
                "ice_fraction0": {"unit": "-", "range": [0.0, 1.0]},
                "dt": {"unit": "s", "range": [1.0, 600.0]},
                "duration_s": {"unit": "s", "range": [60.0, 172800.0]},
            },
            "outputs": {
                "t": "s",
                "enthalpy_J": "J (rel. to liquid at 0 C)",
                "ice_fraction": "-",
                "soc": "-",
                "temperature_C": "degC",
                "UA_eff_W_per_K": "W/K (varies with ice thickness)",
                "ice_radius_m": "m",
                "q_coil_W": "W (>0 freezing, <0 melting)",
                "q_loss_W": "W",
                "cooling_power_W": "W",
                "energy_stored_kwh": "kWh_th",
            },
            "source": self._raw.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    info = m.get_info()
    print(f"{info['component_id']} {info['component_name']} -- {info['fidelity']}")
    # Charge 8 hours at -6 C brine from empty tank.
    r = m.predict({"T_brine_C": -6.0, "T_amb_C": 20.0, "ice_fraction0": 0.0,
                   "dt": 600.0, "duration_s": 8 * 3600.0})
    print(f"After 8 h charge: ice_fraction={r['ice_fraction'][-1]:.3f}, "
          f"T={r['temperature_C'][-1]:.2f} C, "
          f"UA_eff={r['UA_eff_W_per_K'][-1]:.1f} W/K "
          f"(clean {m._model.UA_clean:.1f}), "
          f"stored={r['energy_stored_kwh'][-1]:.1f} kWh_th")
