"""
EC042 -- Pseudocapacitor -- F2a RC-Faradaic (physics-lumped)
Standardised predict() / get_info() interface (mirrors EC001 template).
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import PseudocapacitorF2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for the EC042 F2a physics-lumped pseudocapacitor model."""

    component_id = "EC042"
    component_name = "Pseudocapacitor"
    fidelity = "F2a -- RC-Ladder + Voltage-Dependent Faradaic C(V) + Redox Kinetics + Thermal ODE"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            # allow overriding individual cell/thermal values
            for grp in ("cell", "thermal"):
                if grp in params:
                    self._raw[grp].update(params[grp])
        self._model = PseudocapacitorF2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        Run dynamic galvanostatic simulation.

        inputs:
            current_A   : float or callable(t)  applied current (>0 discharge, <0 charge)
            v_cap0_V    : float  initial capacitor-node voltage (default v_max)
            T0_K        : float  initial temperature (default T_ref)
            dt          : float  output time step (default 0.05 s)
            duration_s  : float  total duration (default 20.0 s)

        returns dict of time-series arrays (see model.simulate).
        """
        I = inputs.get("current_A", 50.0)
        v0 = inputs.get("v_cap0_V", self._model.v_max)
        T0 = inputs.get("T0_K", self._model.T_ref)
        dt = inputs.get("dt", 0.05)
        dur = inputs.get("duration_s", 20.0)
        return self._model.simulate(I, v0, T0, dt, dur)

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "current_A": {"unit": "A", "range": [-100.0, 100.0],
                              "note": ">0 discharge, <0 charge"},
                "v_cap0_V": {"unit": "V", "range": [0.0, 1.0]},
                "T0_K": {"unit": "K", "range": [243.15, 333.15]},
                "dt": {"unit": "s", "range": [0.001, 1.0]},
                "duration_s": {"unit": "s", "range": [0.1, 3600.0]},
            },
            "outputs": {
                "t": "s",
                "v_cap": "V",
                "terminal_voltage": "V",
                "current": "A",
                "power": "W",
                "soc": "-",
                "temperature": "K",
                "stored_energy": "J",
                "capacitance": "F",
                "heat": "W",
                "components": "dict (faradaic_capacitance F, access_factor -)",
            },
            "source": self._raw.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    info = m.get_info()
    print(f"{info['component_id']} {info['component_name']} -- {info['fidelity']}")
    print(f"source: {info['source']}")
    r = m.predict({"current_A": 50.0, "v_cap0_V": 1.0, "duration_s": 3.0, "dt": 0.5})
    print(f"\nGalvanostatic 50 A discharge from 1.0 V:")
    print(f"  V_cap:   {r['v_cap'][0]:.4f} -> {r['v_cap'][-1]:.4f} V")
    print(f"  V_term:  {r['terminal_voltage'][0]:.4f} -> {r['terminal_voltage'][-1]:.4f} V")
    print(f"  C_diff:  {r['capacitance'][0]:.1f} -> {r['capacitance'][-1]:.1f} F")
    print(f"  T:       {r['temperature'][0]:.3f} -> {r['temperature'][-1]:.3f} K")
    print(f"  E_store: {r['stored_energy'][0]:.2f} -> {r['stored_energy'][-1]:.2f} J")
    eta = m._model.round_trip_efficiency(20.0, 0.2, m._model.T_ref)
    print(f"  round-trip efficiency @20 A: {eta:.4f}")
