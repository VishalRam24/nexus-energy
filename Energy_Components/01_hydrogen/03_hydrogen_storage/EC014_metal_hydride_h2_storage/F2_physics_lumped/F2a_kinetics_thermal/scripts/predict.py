"""
EC014 -- Metal Hydride H2 Storage -- F2a Kinetics + Thermal
Standardised predict() / get_info() interface.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import MetalHydrideF2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for the EC014 F2a kinetics + thermal lumped model."""

    component_id = "EC014"
    component_name = "Metal Hydride H2 Storage"
    fidelity = "F2a -- Coupled Reaction Kinetics + Lumped Thermal ODE"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            self._raw["unit"].update(params)
        self._model = MetalHydrideF2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        Run dynamic absorption/desorption simulation.

        inputs:
            P_supply_bar : float (or callable(t))  -- supply pressure [bar]
            T_bed_K      : float  -- initial bed temperature [K]   (default 293.15)
            X0           : float  -- initial H/M loading            (default 0.0)
            dt           : float  -- output step [s]                (default 1.0)
            duration_s   : float  -- total sim time [s]             (default 600.0)
        """
        P = inputs.get("P_supply_bar", 10.0)
        T0 = inputs.get("T_bed_K", 293.15)
        X0 = inputs.get("X0", 0.0)
        dt = inputs.get("dt", 1.0)
        dur = inputs.get("duration_s", 600.0)

        return self._model.simulate(P, T0, X0, dt, dur)

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "P_supply_bar": {"unit": "bar", "range": [0.1, 50.0]},
                "T_bed_K": {"unit": "K", "range": [253.0, 373.0]},
                "X0": {"unit": "atoms_H/formula", "range": [0.0, 6.0]},
                "dt": {"unit": "s"},
                "duration_s": {"unit": "s"},
            },
            "outputs": {
                "t": "s",
                "HM_ratio": "atoms_H/formula",
                "soc": "-",
                "stored_mass_kg": "kg",
                "temperature": "K",
                "P_supply": "bar",
                "P_eq_abs": "bar",
                "P_eq_des": "bar",
                "Q_rxn": "W",
                "Q_cool": "W",
                "gravimetric_wt_pct": "wt%",
            },
            "source": self._raw.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    r = m.predict({"P_supply_bar": 10.0, "T_bed_K": 293.15, "duration_s": 600.0, "dt": 10.0})
    print(
        f"Final H/M: {r['HM_ratio'][-1]:.3f}, SOC: {r['soc'][-1]:.3f}, "
        f"T: {r['temperature'][-1]:.2f} K, stored: {r['stored_mass_kg'][-1]*1000:.2f} g H2"
    )
