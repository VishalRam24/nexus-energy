"""
EC012 -- Compressed Gas H2 Storage -- F2a Thermodynamic Tank
Standardised predict() / get_info() interface.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import CompressedGasH2F2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for EC012 F2a lumped thermodynamic tank model."""

    component_id = "EC012"
    component_name = "Compressed Gas H2 Storage"
    fidelity = "F2a -- Lumped Thermodynamic Tank (open-system first-law ODE, real-gas EOS)"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            self._raw.update(params)
        self._model = CompressedGasH2F2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        Run a fill / discharge simulation of the tank.

        inputs:
            mdot_kg_s   : float or callable(t)  mass flow (>0 fill, <0 discharge)
            T0_K        : float  initial gas/wall temperature (default 298.15)
            T_amb_K     : float  ambient temperature (default param)
            T_in_K      : float  inlet H2 temperature on fill (default param)
            P0_bar      : float  initial pressure (default P_min); used if m0_kg None
            m0_kg       : float  initial gas mass (overrides P0_bar)
            dt          : float  output step [s] (default 1.0)
            duration_s  : float  total time [s] (default 60.0)
        """
        mdot = inputs.get("mdot_kg_s", 0.0)
        T0 = inputs.get("T0_K", 298.15)
        T_amb = inputs.get("T_amb_K", None)
        T_in = inputs.get("T_in_K", None)
        P0 = inputs.get("P0_bar", None)
        m0 = inputs.get("m0_kg", None)
        dt = inputs.get("dt", 1.0)
        dur = inputs.get("duration_s", 60.0)

        return self._model.simulate(mdot, T0, T_amb_K=T_amb, T_in_K=T_in,
                                    m0_kg=m0, P0_bar=P0, dt=dt, duration_s=dur)

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "mdot_kg_s": {"unit": "kg/s", "range": [-0.1, 0.1]},
                "T0_K": {"unit": "K", "range": [200, 373]},
                "T_amb_K": {"unit": "K", "range": [233, 333]},
                "T_in_K": {"unit": "K", "range": [200, 333]},
                "P0_bar": {"unit": "bar", "range": [1, 900]},
                "m0_kg": {"unit": "kg"},
                "dt": {"unit": "s"},
                "duration_s": {"unit": "s"},
            },
            "outputs": {
                "t": "s",
                "mass": "kg",
                "temperature": "K",
                "T_wall": "K",
                "pressure": "bar",
                "density": "kg/m3",
                "soc": "-",
                "energy_MJ": "MJ",
                "mdot": "kg/s",
            },
            "source": self._raw.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    # Fast fill: 0.01 kg/s pre-cooled H2 into a near-empty tank for 120 s
    r = m.predict({"mdot_kg_s": 0.01, "T0_K": 298.15, "P0_bar": 20.0,
                   "dt": 1.0, "duration_s": 120.0})
    print(f"\nFast-fill (0.01 kg/s, 120 s):")
    print(f"  mass  {r['mass'][0]:.3f} -> {r['mass'][-1]:.3f} kg")
    print(f"  P     {r['pressure'][0]:.1f} -> {r['pressure'][-1]:.1f} bar")
    print(f"  T_gas {r['temperature'][0]:.1f} -> {r['temperature'][-1]:.1f} K")
    print(f"  SOC   {r['soc'][0]:.3f} -> {r['soc'][-1]:.3f}")
