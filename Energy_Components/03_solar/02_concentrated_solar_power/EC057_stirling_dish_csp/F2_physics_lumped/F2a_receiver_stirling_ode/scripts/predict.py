"""
EC057 -- Stirling Dish CSP -- F2a Physics-Lumped (Receiver ODE + Stirling Engine)
Standardised predict() / get_info() interface (mirrors EC001 template).
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import StirlingDishF2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for EC057 F2a transient dish-Stirling model."""

    component_id = "EC057"
    component_name = "Stirling Dish CSP"
    fidelity = "F2a -- Lumped Receiver Thermal ODE + Carnot-Limited Stirling Engine"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            self._raw["unit"].update(params)
        self._model = StirlingDishF2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        Run the transient receiver/engine simulation.

        inputs:
            DNI          : float (or callable(t))  Direct Normal Irradiance [W/m2]
            theta        : float   residual tracking incidence angle [deg]   (default 0)
            T_rec_init   : float   initial receiver wall temperature [degC]  (default 25)
            T_amb        : float   ambient temperature [degC]                (default 25)
            dt           : float   output time step [s]                      (default 5.0)
            duration_s   : float   total simulation duration [s]             (default 1800)
        """
        dni = inputs.get("DNI", 900.0)
        theta = inputs.get("theta", 0.0)
        T_rec_init = inputs.get("T_rec_init", 25.0)
        T_amb = inputs.get("T_amb", 25.0)
        dt = inputs.get("dt", 5.0)
        dur = inputs.get("duration_s", 1800.0)

        return self._model.simulate(dni, theta, T_rec_init, T_amb, dt, dur)

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "DNI": {"unit": "W/m2", "range": [0, 1100]},
                "theta": {"unit": "deg", "range": [0, 10]},
                "T_rec_init": {"unit": "degC", "range": [20, 850]},
                "T_amb": {"unit": "degC", "range": [-10, 45]},
                "dt": {"unit": "s"},
                "duration_s": {"unit": "s"},
            },
            "outputs": {
                "t": "s",
                "T_rec_c": "degC (receiver wall temperature)",
                "P_elec_kw": "kW (net electrical output)",
                "Q_absorbed_kw": "kW",
                "Q_loss_kw": "kW",
                "Q_rad_kw": "kW",
                "Q_engine_kw": "kW",
                "eta_carnot": "-",
                "eta_stirling": "-",
                "eta_system": "-",
            },
            "source": self._raw["unit"].get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    r = m.predict({"DNI": 900.0, "T_rec_init": 25.0, "T_amb": 25.0,
                   "dt": 10.0, "duration_s": 1800.0})
    print(f"\nWarm-up to steady state @ DNI=900 W/m2:")
    print(f"  Final receiver T : {r['T_rec_c'][-1]:.1f} degC")
    print(f"  Final P_elec     : {r['P_elec_kw'][-1]:.2f} kW")
    print(f"  Final eta_system : {r['eta_system'][-1]*100:.1f} %")
    print(f"  eta_stirling     : {r['eta_stirling'][-1]*100:.1f} % "
          f"(Carnot {r['eta_carnot'][-1]*100:.1f} %)")
