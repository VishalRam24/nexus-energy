"""
EC037 -- Zinc-Bromine Flow Battery (ZBFB) -- F2a Physics-Lumped Stack Model
Standardised predict() / get_info() interface (mirrors EC001 F2a template).
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import ZnBrFlowF2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for the ZBFB F2a physics-lumped stack model."""

    component_id = "EC037"
    component_name = "Zinc-Bromine Flow Battery (ZBFB)"
    fidelity = "F2a -- Physics-Lumped Stack (Nernst + overpotentials + SOC/Br2/thermal ODE)"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            self._raw["unit"].update(params)
        self._model = ZnBrFlowF2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        Run a dynamic stack simulation.

        inputs:
            current_A   : float or list -- stack current [A], +discharge (default 100)
            soc0        : float -- initial SOC (default 0.5)
            T0          : float -- initial temperature [K] (default 298.15)
            flow_Lpm    : float -- electrolyte flow [L/min] (default 2.0)
            c_Br2_0     : float -- initial Br2 conc [mol/L] (default soc0*c_max)
            dt          : float -- output time step [s] (default 1.0)
            duration_s  : float -- simulation duration [s] (default 600.0)
        """
        return self._model.simulate(
            current_A=inputs.get("current_A", 100.0),
            soc0=inputs.get("soc0", 0.5),
            T0=inputs.get("T0", 298.15),
            flow_Lpm=inputs.get("flow_Lpm", 2.0),
            c_Br2_0=inputs.get("c_Br2_0", None),
            dt=inputs.get("dt", 1.0),
            duration_s=inputs.get("duration_s", 600.0),
        )

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "current_A": {"unit": "A", "range": [-300, 300], "note": "+discharge"},
                "soc0": {"unit": "-", "range": [0.05, 0.95]},
                "T0": {"unit": "K", "range": [288.15, 313.15]},
                "flow_Lpm": {"unit": "L/min", "range": [0.2, 8.0]},
                "c_Br2_0": {"unit": "mol/L"},
                "dt": {"unit": "s"},
                "duration_s": {"unit": "s"},
            },
            "outputs": {
                "t": "s",
                "soc": "-",
                "c_Br2": "mol/L",
                "temperature": "K",
                "voltage": "V (stack)",
                "ocv": "V (stack)",
                "current": "A",
                "coulombic_efficiency": "-",
                "shuttle_current": "A",
                "power": "W",
            },
            "source": self._raw.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    r = m.predict({"current_A": -100.0, "soc0": 0.3, "duration_s": 300.0, "dt": 10.0})
    print(
        f"Charge sim: SOC {r['soc'][0]:.3f}->{r['soc'][-1]:.3f}, "
        f"V {r['voltage'][-1]:.2f} V, T {r['temperature'][-1]:.2f} K, "
        f"eta_C {r['coulombic_efficiency'][-1]:.3f}"
    )
