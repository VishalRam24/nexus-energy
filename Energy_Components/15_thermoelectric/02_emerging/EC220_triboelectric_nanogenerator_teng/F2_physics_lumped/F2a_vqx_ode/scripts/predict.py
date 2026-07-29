"""
EC220 -- Triboelectric Nanogenerator (TENG) -- F2a Physics-Lumped V-Q-x Model
Standardised predict() / get_info() interface (mirrors EC001 template).
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import TENG_F2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for TENG F2a physics-lumped V-Q-x ODE model."""

    component_id = "EC220"
    component_name = "Triboelectric Nanogenerator (TENG)"
    fidelity = "F2a -- Physics-Lumped V-Q-x Governing Equation with Charge ODE"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            self._raw["unit"].update(params)
        self._model = TENG_F2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        Run the TENG charge-ODE simulation and optionally sweep load/frequency.

        inputs:
            frequency_hz : float (default 3.0)        contact-separation frequency
            R_load_ohm   : float (default 1e7)        external resistive load
            n_cycles     : int   (default 5)          cycles to integrate
            sweep_load   : bool  (default False)      include power-vs-load + optimum
            sweep_freq   : bool  (default False)      include power-vs-frequency
        """
        f = inputs.get("frequency_hz", 3.0)
        R = inputs.get("R_load_ohm", 1e7)
        n_cycles = inputs.get("n_cycles", 5)

        result = self._model.simulate(f, R, n_cycles)

        if inputs.get("sweep_load", False):
            R_arr, P_arr, R_opt = self._model.power_vs_load(f)
            result["sweep_R_load"] = R_arr
            result["sweep_power_vs_load"] = P_arr
            result["R_optimal_ohm"] = R_opt

        if inputs.get("sweep_freq", False):
            f_arr, P_arr = self._model.power_vs_frequency(R_load_ohm=R)
            result["sweep_frequency"] = f_arr
            result["sweep_power_vs_freq"] = P_arr

        return result

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "frequency_hz": {"unit": "Hz", "range": [0.1, 100.0]},
                "R_load_ohm": {"unit": "ohm", "range": [1e4, 1e10]},
                "n_cycles": {"unit": "-", "range": [1, 200]},
                "sweep_load": {"unit": "bool"},
                "sweep_freq": {"unit": "bool"},
            },
            "outputs": {
                "t": "s",
                "charge": "C",
                "voltage": "V",
                "current": "A",
                "power": "W",
                "gap": "m",
                "energy_per_cycle": "J",
                "power_avg": "W",
                "power_density_mwcm2": "mW/cm2",
                "R_optimal_ohm": "ohm (if sweep_load)",
            },
            "source": self._raw.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    r = m.predict({"frequency_hz": 3.0, "R_load_ohm": 1e7, "n_cycles": 5,
                   "sweep_load": True})
    print(f"V_peak = {r['V_peak']:.2f} V, I_peak = {r['I_peak']*1e6:.3f} uA")
    print(f"Avg power = {r['power_avg']*1e6:.4f} uW, "
          f"power density = {r['power_density_mwcm2']*1e3:.4f} uW/cm2")
    print(f"Energy/cycle = {r['energy_per_cycle']*1e6:.4f} uJ")
    print(f"Net charge/cycle = {r['net_charge_per_cycle']:.3e} C (should ~0)")
    print(f"Optimal load = {r['R_optimal_ohm']:.3e} ohm")
