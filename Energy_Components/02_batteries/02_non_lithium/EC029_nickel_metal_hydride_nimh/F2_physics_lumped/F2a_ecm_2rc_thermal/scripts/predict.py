"""
EC029 -- NiMH Battery -- F2a Thevenin 2-RC Electrothermal
Standardised predict() / get_info() interface (mirrors EC001 template).
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import NiMH_F2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for the NiMH F2a Thevenin 2-RC electrothermal model."""

    component_id = "EC029"
    component_name = "Nickel-Metal Hydride (NiMH) Battery"
    fidelity = "F2a -- Thevenin 2-RC ECM with Coulomb-counted SOC + exothermic-overcharge thermal ODE"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            self._raw["unit"].update(params)
        self._model = NiMH_F2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        Run dynamic simulation.

        inputs:
            current_A   : float or callable(t)  (I>0 discharge, I<0 charge)
            soc0        : float  (default 0.5)
            T0_K        : float  (default ambient 298.15)
            dt          : float  (default 1.0)
            duration_s  : float  (default 600.0)
        """
        I = inputs.get("current_A", 2.0)
        soc0 = inputs.get("soc0", 0.5)
        T0 = inputs.get("T0_K", None)
        dt = inputs.get("dt", 1.0)
        dur = inputs.get("duration_s", 600.0)
        return self._model.simulate(I, soc0=soc0, T0=T0, dt=dt, duration_s=dur)

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "current_A": {"unit": "A", "range": [-20.0, 20.0], "note": "I>0 discharge, I<0 charge"},
                "soc0": {"unit": "-", "range": [0.0, 1.05]},
                "T0_K": {"unit": "K", "range": [253.15, 333.15]},
                "dt": {"unit": "s", "range": [0.1, 10.0]},
                "duration_s": {"unit": "s", "range": [1.0, 36000.0]},
            },
            "outputs": {
                "t": "s",
                "soc": "-",
                "voltage": "V",
                "current": "A",
                "power": "W",
                "temperature": "K",
                "efficiency": "-",
                "overcharge_fraction": "-",
                "heat": "dict of W arrays (irreversible, reversible, recombination, loss)",
            },
            "source": self._raw.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    # 2 A discharge for 5 min
    r = m.predict({"current_A": 2.0, "soc0": 0.8, "dt": 5.0, "duration_s": 300.0})
    print(f"Discharge -> Final SOC: {r['soc'][-1]:.4f}, "
          f"V: {r['voltage'][-1]:.4f} V, T: {r['temperature'][-1]:.2f} K")
    # overcharge demo: charge at -3 A starting near full
    r2 = m.predict({"current_A": -3.0, "soc0": 0.93, "dt": 5.0, "duration_s": 600.0})
    print(f"Overcharge -> Final SOC: {r2['soc'][-1]:.4f}, "
          f"V: {r2['voltage'][-1]:.4f} V, dT: {r2['temperature'][-1]-r2['temperature'][0]:.2f} K, "
          f"f_oc: {r2['overcharge_fraction'][-1]:.3f}")
