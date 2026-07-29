"""
EC223 -- Radioisotope Thermoelectric Generator (RTG) -- F2a Physics-Lumped
Standardised predict() / get_info() interface (mirrors EC001 template).
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import RTG_F2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for RTG F2a decay-heat + thermoelectric ODE model."""

    component_id = "EC223"
    component_name = "Radioisotope Thermoelectric Generator (RTG)"
    fidelity = "F2a -- Physics-Lumped Decay-Heat + Thermoelectric Module with Hot-Side Thermal ODE"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            self._raw["unit"].update(params)
        self._model = RTG_F2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        Run the mission-lifetime hot-side ODE simulation.

        inputs:
            mission_years : float (default 50.0)
            n_points      : int   (default 200)
            T_h0          : float initial hot-side temperature [K] (default BOL)
            R_load        : float load resistance per couple [ohm] (default matched)

        returns dict of arrays: t_years, T_hot_K, Q_decay_W, P_electric_W,
            eta_module, eta_carnot, eta_zt_max, current_A, Q_radiator_W,
            power_fraction
        """
        mission = inputs.get("mission_years", 50.0)
        n_points = int(inputs.get("n_points", 200))
        T_h0 = inputs.get("T_h0", None)
        R_load = inputs.get("R_load", None)
        return self._model.simulate(mission, n_points, T_h0=T_h0, R_load=R_load)

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "mission_years": {"unit": "years", "range": [0, 200]},
                "n_points": {"unit": "-", "range": [2, 5000]},
                "T_h0": {"unit": "K", "range": [600, 1400]},
                "R_load": {"unit": "ohm/couple", "range": [0.1e-2, 100.0]},
            },
            "outputs": {
                "t_years": "years",
                "T_hot_K": "K",
                "Q_decay_W": "W (thermal)",
                "P_electric_W": "W (electrical)",
                "eta_module": "-",
                "eta_carnot": "-",
                "eta_zt_max": "-",
                "current_A": "A",
                "Q_radiator_W": "W",
                "power_fraction": "-",
            },
            "source": self._raw.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    r = m.predict({"mission_years": 50.0, "n_points": 51})
    print(f"BOL : T_hot={r['T_hot_K'][0]:.1f} K  P_e={r['P_electric_W'][0]:.1f} W  "
          f"eta={r['eta_module'][0]*100:.2f}%  (Carnot {r['eta_carnot'][0]*100:.1f}%)")
    print(f"50yr: T_hot={r['T_hot_K'][-1]:.1f} K  P_e={r['P_electric_W'][-1]:.1f} W  "
          f"power_fraction={r['power_fraction'][-1]*100:.1f}%")
