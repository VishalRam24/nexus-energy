"""
EC059 — Evacuated Tube Solar Collector — F2a Lumped-Capacitance
Standardised predict() / get_info() interface (mirrors EC001 template).
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import EvacuatedTubeF2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for EC059 F2a lumped-capacitance dynamic model."""

    component_id = "EC059"
    component_name = "Evacuated Tube Solar Collector"
    fidelity = "F2a — Lumped-Capacitance Dynamic Energy Balance ODE"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            self._raw["unit"].update(params)
        self._model = EvacuatedTubeF2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        Run dynamic simulation of the evacuated tube collector.

        inputs:
            irradiance     : float or callable f(t)->W/m2   (default 800)
            T_ambient_c    : float or callable              (default 20.0)
            T_inlet_c      : float or callable              (default param T_in_default)
            theta_deg      : float or callable              (default 0.0)
            T0_c           : float (initial absorber temp)  (default = ambient)
            dt             : float [s]                       (default 10.0)
            duration_s     : float [s]                       (default 3600.0)

        returns dict of time-series arrays (t, T_absorber_c, T_outlet_c,
            useful_heat_w, efficiency, q_absorbed_w, q_loss_w, U_L_w_m2k, ...).
        """
        G = inputs.get("irradiance", 800.0)
        Ta = inputs.get("T_ambient_c", 20.0)
        Tin = inputs.get("T_inlet_c", None)
        theta = inputs.get("theta_deg", 0.0)
        T0 = inputs.get("T0_c", None)
        dt = inputs.get("dt", 10.0)
        dur = inputs.get("duration_s", 3600.0)

        return self._model.simulate(G, Ta, Tin, theta, T0, dt, dur)

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "irradiance": {"unit": "W/m2", "range": [0, 1200]},
                "T_ambient_c": {"unit": "degC", "range": [-20, 45]},
                "T_inlet_c": {"unit": "degC", "range": [10, 180]},
                "theta_deg": {"unit": "deg", "range": [0, 80]},
                "T0_c": {"unit": "degC"},
                "dt": {"unit": "s", "range": [1, 600]},
                "duration_s": {"unit": "s", "range": [1, 86400]},
            },
            "outputs": {
                "t": "s",
                "T_absorber_c": "degC",
                "T_outlet_c": "degC",
                "useful_heat_w": "W",
                "efficiency": "-",
                "q_absorbed_w": "W",
                "q_loss_w": "W",
                "U_L_w_m2k": "W/m2K",
                "reduced_temp": "(Tm-Ta)/G  [m2K/W]",
            },
            "source": self._raw.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    r = m.predict({"irradiance": 800.0, "T_ambient_c": 20.0,
                   "T_inlet_c": 40.0, "duration_s": 3600.0, "dt": 30.0})
    print(f"\nSteady-state absorber T: {r['T_absorber_c'][-1]:.2f} C")
    print(f"Outlet T: {r['T_outlet_c'][-1]:.2f} C")
    print(f"Useful heat: {r['useful_heat_w'][-1]:.1f} W")
    print(f"Efficiency: {r['efficiency'][-1]:.4f}")
    print(f"U_L: {r['U_L_w_m2k'][-1]:.3f} W/m2K")
