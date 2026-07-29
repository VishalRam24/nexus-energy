"""
EC013 -- Liquid Hydrogen (LH2) Storage -- F2a Two-Phase Lumped Cryogenic Tank
Standardised predict() / get_info() interface (mirrors EC001 F2a).
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import LH2TwoPhaseTank

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for LH2 storage F2a two-phase cryogenic tank model."""

    component_id = "EC013"
    component_name = "Liquid Hydrogen (LH2) Storage"
    fidelity = "F2a -- Two-Phase Lumped Cryogenic Tank (self-pressurization + boil-off ODE)"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            self._raw["unit"].update(params)
        self._model = LH2TwoPhaseTank(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        Run dynamic two-phase tank simulation.

        inputs:
            fill_fraction : float  (initial liquid fill, default 0.90)
            T_ambient_K   : float  (ambient temperature, default 298.15)
            P0_bar        : float  (initial pressure, default 1.01325)
            duration_s    : float  (default 86400 = 1 day)
            n_steps       : int    (default 400)
            sealed        : bool   (True = closed/self-pressurizing,
                                    False = vent open / NBP boil-off; default True)
        """
        f      = inputs.get("fill_fraction", 0.90)
        T_amb  = inputs.get("T_ambient_K", 298.15)
        P0     = inputs.get("P0_bar", 1.01325)
        dur    = inputs.get("duration_s", 86400.0)
        nsteps = inputs.get("n_steps", 400)
        sealed = inputs.get("sealed", True)

        return self._model.simulate(f, T_amb, P0, dur, nsteps, sealed)

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "fill_fraction": {"unit": "-", "range": [0.05, 0.95]},
                "T_ambient_K": {"unit": "K", "range": [233.15, 333.15]},
                "P0_bar": {"unit": "bar", "range": [1.0, 6.0]},
                "duration_s": {"unit": "s", "range": [60.0, 2592000.0]},
                "n_steps": {"unit": "-"},
                "sealed": {"unit": "bool"},
            },
            "outputs": {
                "t": "s",
                "m_liquid": "kg",
                "m_vapor": "kg",
                "m_total": "kg",
                "temperature": "K",
                "pressure": "bar",
                "heat_leak_W": "W",
                "boiloff_rate_kg_s": "kg/s",
                "BOR_pct_day": "%/day",
                "fill_fraction": "-",
                "energy_stored_MJ": "MJ",
            },
            "source": self._raw.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    # 7-day sealed dormancy of a 90%-full 1 m3 dewar
    r = m.predict({"fill_fraction": 0.90, "duration_s": 7 * 86400.0, "sealed": True})
    print(f"\nSealed 7-day dormancy (90% fill, T_amb=298 K):")
    print(f"  P:   {r['pressure'][0]:.3f} -> {r['pressure'][-1]:.3f} bar")
    print(f"  T:   {r['temperature'][0]:.3f} -> {r['temperature'][-1]:.3f} K")
    print(f"  m_L: {r['m_liquid'][0]:.3f} -> {r['m_liquid'][-1]:.3f} kg")
    print(f"  BOR(open-vent, day1): {r['BOR_pct_day'][0]:.4f} %/day")
    # open-vent boil-off
    r2 = m.predict({"fill_fraction": 0.90, "duration_s": 86400.0, "sealed": False})
    print(f"\nOpen-vent 1-day boil-off: BOR={r2['BOR_pct_day'][0]:.4f} %/day, "
          f"vented {r2['m_total'][0]-r2['m_total'][-1]:.4f} kg")
