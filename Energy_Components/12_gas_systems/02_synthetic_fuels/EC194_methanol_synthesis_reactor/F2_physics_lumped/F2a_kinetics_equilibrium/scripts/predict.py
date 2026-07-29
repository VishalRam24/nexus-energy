"""
EC194 -- Methanol Synthesis Reactor -- F2a Kinetics + Equilibrium
Standardised predict() / get_info() interface.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import MethanolReactor_F2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for Methanol Synthesis Reactor F2a CSTR model."""

    component_id = "EC194"
    component_name = "Methanol Synthesis Reactor"
    fidelity = "F2a -- Kinetics + Equilibrium CSTR Model"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            self._raw["unit"].update(params)
        self._model = MethanolReactor_F2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        Run methanol-synthesis CSTR dynamic simulation.

        inputs:
            T0_K : float          Initial reactor temperature [K] (default = T_in)
            duration_s : float    Simulation duration [s] (default 600)
            dt : float            Output time step [s] (default 1.0)
            P_bar : float         Total pressure [bar] (default from params, 50)
            GHSV : float          Gas hourly space velocity [1/h] (default 10000)
            T_in_K : float        Feed inlet temperature [K] (default from params)
            T_cool_K : float      Coolant temperature [K] (default from params)
        """
        T0 = inputs.get("T0_K", self._model.T_in)
        dur = inputs.get("duration_s", 600.0)
        dt = inputs.get("dt", 1.0)
        P = inputs.get("P_bar", None)
        GHSV = inputs.get("GHSV", None)
        T_in = inputs.get("T_in_K", None)
        T_cool = inputs.get("T_cool_K", None)

        return self._model.simulate(
            T0=T0, duration_s=dur, dt=dt, P=P, GHSV=GHSV,
            T_in=T_in, T_cool=T_cool,
        )

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "T0_K": {"unit": "K", "range": [473.15, 573.15]},
                "duration_s": {"unit": "s"},
                "dt": {"unit": "s"},
                "P_bar": {"unit": "bar", "range": [30.0, 100.0]},
                "GHSV": {"unit": "1/h", "range": [2000, 20000]},
                "T_in_K": {"unit": "K", "range": [473.15, 573.15]},
                "T_cool_K": {"unit": "K", "range": [490.0, 540.0]},
            },
            "outputs": {
                "t": "s",
                "T": "K",
                "X_C": "- (per-pass CO_x carbon conversion to MeOH)",
                "meoh_yield": "- (mol MeOH per mol carbon fed)",
                "y_MeOH_dry": "- (dry MeOH mole fraction)",
                "y_MeOH_wet": "- (wet MeOH mole fraction)",
                "C_CO": "mol/m3", "C_CO2": "mol/m3", "C_H2": "mol/m3",
                "C_CH3OH": "mol/m3", "C_H2O": "mol/m3",
                "T_max": "K",
                "thermal_runaway": "bool",
                "X_eq_final": "- (equilibrium conversion at final T)",
            },
            "source": self._raw.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    r = m.predict({"duration_s": 400.0, "dt": 5.0})
    print(f"Final T:                {r['T'][-1]:.1f} K")
    print(f"Final carbon conversion:{r['X_C'][-1]:.4f}")
    print(f"Final MeOH dry fraction:{r['y_MeOH_dry'][-1]:.4f}")
    print(f"Equilibrium conversion: {r['X_eq_final']:.4f}")
    print(f"Thermal runaway:        {r['thermal_runaway']}")
