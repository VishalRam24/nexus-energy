"""
EC038 -- Iron-Chromium Flow Battery (ICFB) -- F2a Physics-Lumped Stack Model
Standardised predict() / get_info() interface (mirrors EC001 template).
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import FeCrFlowBatteryF2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for the Fe-Cr flow battery F2a physics-lumped model."""

    component_id = "EC038"
    component_name = "Iron-Chromium Flow Battery (ICFB)"
    fidelity = "F2a -- Physics-Lumped Stack Model (Nernst + Butler-Volmer + thermal/SOC ODE)"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            self._raw["unit"].update(params)
        self._model = FeCrFlowBatteryF2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        Run dynamic simulation.

        inputs:
            current_A   : float (or callable) -- stack current [A]; >0 discharge, <0 charge
            soc0        : float  initial SOC [0,1]   (default 0.5)
            T0          : float  initial temperature [K] (default 308.15)
            Q_flow      : float  electrolyte flow per side [m3/s] (default nominal)
            dt          : float  output time step [s]  (default 10.0)
            duration_s  : float  total duration [s]    (default 3600.0)
        """
        I = inputs.get("current_A", 50.0)
        soc0 = inputs.get("soc0", 0.5)
        T0 = inputs.get("T0", 308.15)
        Q_flow = inputs.get("Q_flow", None)
        dt = inputs.get("dt", 10.0)
        dur = inputs.get("duration_s", 3600.0)
        return self._model.simulate(I, soc0, T0, dt, dur, Q_flow)

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "current_A": {"unit": "A", "range": [-200, 200], "note": ">0 discharge, <0 charge"},
                "soc0": {"unit": "-", "range": [0.05, 0.95]},
                "T0": {"unit": "K", "range": [288.15, 338.15]},
                "Q_flow": {"unit": "m3/s"},
                "dt": {"unit": "s"},
                "duration_s": {"unit": "s"},
            },
            "outputs": {
                "t": "s",
                "soc": "-",
                "temperature": "K",
                "voltage": "V (stack terminal)",
                "power": "W",
                "ocv": "V (stack open-circuit)",
                "efficiency": "- (voltage efficiency)",
                "coulombic_eff": "- (charge, accounts for H2)",
                "I_H2": "A (parasitic H2 current)",
                "overpotentials": "dict of V arrays",
            },
            "source": self._raw.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    info = m.get_info()
    print(f"{info['component_id']} {info['component_name']} -- {info['fidelity']}")
    # charge for 1 hour at 50 A
    r = m.predict({"current_A": -50.0, "soc0": 0.4, "T0": 308.15,
                   "dt": 60.0, "duration_s": 3600.0})
    print(f"Charge 50A/1h: SOC {r['soc'][0]:.3f} -> {r['soc'][-1]:.3f}, "
          f"V {r['voltage'][-1]:.1f} V, T {r['temperature'][-1]:.2f} K, "
          f"coulombic_eff {r['coulombic_eff'][-1]:.4f}, I_H2 {r['I_H2'][-1]:.2f} A")
    # discharge
    r2 = m.predict({"current_A": 50.0, "soc0": 0.6, "T0": 308.15,
                    "dt": 60.0, "duration_s": 3600.0})
    print(f"Discharge 50A/1h: SOC {r2['soc'][0]:.3f} -> {r2['soc'][-1]:.3f}, "
          f"V {r2['voltage'][-1]:.1f} V, eff {r2['efficiency'][-1]:.4f}")
