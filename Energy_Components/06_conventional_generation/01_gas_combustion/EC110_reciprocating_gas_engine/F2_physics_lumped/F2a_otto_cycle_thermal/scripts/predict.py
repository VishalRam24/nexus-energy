"""
EC110 -- Reciprocating Gas Engine -- F2a Otto/Miller Cycle + Thermal ODE
Standardised predict() / get_info() interface (mirrors EC001 template).
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import ReciprocatingGasEngineF2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for EC110 F2a Otto/Miller cycle physics-lumped model."""

    component_id = "EC110"
    component_name = "Reciprocating Gas Engine"
    fidelity = "F2a -- Otto/Miller Cycle Thermodynamics with Engine-Block Thermal ODE"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            self._raw["unit"].update(params)
        self._model = ReciprocatingGasEngineF2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        Run engine-block thermal transient + per-step performance.

        inputs:
            part_load_ratio : float (or callable t->PLR for a load schedule)
            T0_K            : float  (initial block temperature, default 298.15)
            speed_rpm       : float  (default rated speed)
            lambda_excess_air : float (default lean-burn lambda)
            dt              : float  (default 1.0 s)
            duration_s      : float  (default 600.0 s)
        """
        plr = inputs.get("part_load_ratio", 1.0)
        T0 = inputs.get("T0_K", 298.15)
        speed = inputs.get("speed_rpm", None)
        lam = inputs.get("lambda_excess_air", None)
        dt = inputs.get("dt", 1.0)
        dur = inputs.get("duration_s", 600.0)

        result = self._model.simulate(plr, T0, speed, lam, dt, dur)
        # Append a steady-state operating-point snapshot at final load
        plr_final = plr(dur) if callable(plr) else plr
        result["operating_point"] = self._model.operating_point(plr_final, speed, lam)
        return result

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "part_load_ratio": {"unit": "-", "range": [0.4, 1.0]},
                "T0_K": {"unit": "K", "range": [253.15, 373.15]},
                "speed_rpm": {"unit": "rpm", "range": [900, 1800]},
                "lambda_excess_air": {"unit": "-", "range": [1.0, 2.2]},
                "dt": {"unit": "s"},
                "duration_s": {"unit": "s"},
            },
            "outputs": {
                "t": "s",
                "temperature": "K (engine block)",
                "P_brake_w": "W",
                "P_fuel_w": "W",
                "eta_brake": "-",
                "eta_indicated": "-",
                "eta_otto": "-",
                "operating_point": "dict of powers/efficiencies/meps",
            },
            "source": self._raw.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    r = m.predict({"part_load_ratio": 1.0, "duration_s": 600.0, "dt": 10.0})
    op = r["operating_point"]
    print(
        f"Rated: P_brake={op['P_brake_w']/1e3:.1f} kW, "
        f"eta_brake={op['eta_brake']*100:.1f}%, "
        f"eta_ind={op['eta_indicated']*100:.1f}%, "
        f"eta_otto={op['eta_otto']*100:.1f}%"
    )
    print(f"Final block T: {r['temperature'][-1]:.2f} K")
