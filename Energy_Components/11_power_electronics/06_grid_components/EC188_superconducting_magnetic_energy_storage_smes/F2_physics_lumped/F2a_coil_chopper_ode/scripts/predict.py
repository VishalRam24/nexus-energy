"""
EC188 -- Superconducting Magnetic Energy Storage (SMES) -- F2a Physics-Lumped
Standardised predict() / get_info() interface.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import SMES_F2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for SMES F2a coil-current ODE + chopper model."""

    component_id = "EC188"
    component_name = "Superconducting Magnetic Energy Storage (SMES)"
    fidelity = "F2a -- Physics-Lumped Coil-Current ODE with Chopper + Cryo Load"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            self._raw["unit"].update(params)
        self._model = SMES_F2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        Run a dynamic coil simulation.

        inputs:
            I0_A        : initial coil current [A]            (default 0.0)
            command     : V_chop [V] (voltage mode) or P_req DC coil power [W]
                          (power mode). Positive => charging.  (default 2500.0)
            mode        : "voltage" or "power"                (default "voltage")
            dt          : output step [s]                     (default 0.01)
            duration_s  : total time [s]                      (default 1.0)
        """
        I0 = inputs.get("I0_A", 0.0)
        command = inputs.get("command", 2500.0)
        mode = inputs.get("mode", "voltage")
        dt = inputs.get("dt", 0.01)
        dur = inputs.get("duration_s", 1.0)
        return self._model.simulate(I0, command, mode=mode, dt=dt, duration_s=dur)

    def round_trip(self, P_MW=None) -> dict:
        """Convenience: full charge/discharge round-trip efficiency."""
        P_W = None if P_MW is None else P_MW * 1e6
        return self._model.round_trip_efficiency(P_W=P_W)

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "I0_A": {"unit": "A", "range": [0.0, self._model.I_max]},
                "command": {"unit": "V or W", "note": "voltage or DC coil power; + = charge"},
                "mode": {"values": ["voltage", "power"]},
                "dt": {"unit": "s"},
                "duration_s": {"unit": "s"},
            },
            "outputs": {
                "t": "s",
                "I_coil_A": "A",
                "V_chop_V": "V",
                "E_stored_MJ": "MJ",
                "SOC": "-",
                "P_coil_MW": "MW (signed, + into coil)",
                "P_grid_MW": "MW (signed, + drawn from grid)",
                "P_cryo_MW": "MW",
            },
            "key_params": {
                "L_H": self._model.L,
                "I_max_A": self._model.I_max,
                "E_max_MJ": self._model.E_max_MJ,
                "P_rated_MW": self._model.P_rated / 1e6,
                "eta_converter": self._model.eta_conv,
                "P_cryo_MW": self._model.P_cryo / 1e6,
            },
            "source": self._raw["unit"].get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    info = m.get_info()
    print(info["component_id"], info["component_name"], "|", info["fidelity"])
    print("E_max =", info["key_params"]["E_max_MJ"], "MJ")

    # Charge from empty at constant chopper voltage for 1 s.
    r = m.predict({"I0_A": 0.0, "command": 2500.0, "mode": "voltage",
                   "dt": 0.01, "duration_s": 1.0})
    print(f"Charge 1 s @2500 V: I {r['I_coil_A'][0]:.1f} -> {r['I_coil_A'][-1]:.1f} A, "
          f"E {r['E_stored_MJ'][0]:.3f} -> {r['E_stored_MJ'][-1]:.3f} MJ")

    rt = m.round_trip(P_MW=1.0)
    print(f"Round-trip eff @1 MW: {rt['eta_rt']*100:.2f} %  "
          f"(in {rt['E_grid_in_MJ']:.2f} MJ, out {rt['E_grid_out_MJ']:.2f} MJ)")
