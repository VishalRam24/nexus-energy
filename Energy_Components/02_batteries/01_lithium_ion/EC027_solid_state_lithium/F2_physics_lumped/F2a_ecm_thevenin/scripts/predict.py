"""
EC027 -- Solid-State Lithium Battery -- F2a Thevenin ECM
Standardised predict() / get_info() interface (mirrors EC001 template).
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import SolidStateLiECM_F2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for EC027 F2a Thevenin equivalent-circuit model."""

    component_id = "EC027"
    component_name = "Solid-State Lithium Battery"
    fidelity = "F2a -- Thevenin ECM (1-RC/2-RC) with Solid-Electrolyte Arrhenius R(T) and Thermal ODE"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            # shallow per-section override
            for k, v in params.items():
                if isinstance(v, dict) and k in self._raw and isinstance(self._raw[k], dict):
                    self._raw[k].update(v)
                else:
                    self._raw[k] = v
        self._model = SolidStateLiECM_F2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        Run a dynamic ECM + thermal simulation.

        inputs:
            current_A   : float or callable(t)  load current (>0 discharge, <0 charge)
            soc0        : float  initial SOC (default 0.8)
            T0          : float  initial cell temperature K (default 298.15)
            T_amb       : float  ambient temperature K (default param)
            dt          : float  output step s (default 1.0)
            duration_s  : float  total duration s (default 600.0)
        """
        return self._model.simulate(
            current_A=inputs.get("current_A", 4.0),
            soc0=inputs.get("soc0", 0.8),
            T0=inputs.get("T0", 298.15),
            T_amb=inputs.get("T_amb", None),
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
                "current_A": {"unit": "A", "range": [-20.0, 20.0], "note": ">0 discharge, <0 charge"},
                "soc0": {"unit": "-", "range": [0.0, 1.0]},
                "T0": {"unit": "K", "range": [263.15, 333.15]},
                "T_amb": {"unit": "K", "range": [263.15, 333.15]},
                "dt": {"unit": "s"},
                "duration_s": {"unit": "s"},
            },
            "outputs": {
                "t": "s",
                "soc": "-",
                "voltage": "V",
                "current": "A",
                "power": "W",
                "temperature": "K",
                "R0": "Ohm (series SE ionic resistance)",
                "ocv": "V",
                "v_rc": "list of V arrays (RC branch voltages)",
                "coulombic_efficiency": "- (0,1)",
            },
            "source": self._raw.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    r = m.predict({"current_A": 4.0, "soc0": 0.9, "T0": 298.15, "dt": 5.0, "duration_s": 600.0})
    print(f"Final SOC: {r['soc'][-1]:.4f}, Final V: {r['voltage'][-1]:.4f} V, "
          f"Final T: {r['temperature'][-1]:.2f} K, "
          f"R0(end): {r['R0'][-1]*1000:.2f} mOhm, "
          f"coulombic_eff: {r['coulombic_efficiency']:.4f}")
