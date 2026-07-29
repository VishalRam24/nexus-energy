"""
EC193 -- Methanation Reactor (Power-to-Gas) -- F2a Kinetics + Equilibrium
Standardised predict() / get_info() interface.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import MethanationReactor_F2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for Methanation Reactor F2a CSTR model."""

    component_id = "EC193"
    component_name = "Methanation Reactor (Power-to-Gas)"
    fidelity = "F2a -- Kinetics + Equilibrium CSTR Model"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            self._raw["unit"].update(params)
        self._model = MethanationReactor_F2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        Run CSTR dynamic simulation.

        inputs:
            T0_K : float          Initial/inlet temperature [K] (default 523.15)
            duration_s : float    Simulation duration [s] (default 600)
            dt : float            Output time step [s] (default 1.0)
            P_bar : float         Total pressure [bar] (default 10)
            GHSV : float          Gas hourly space velocity [1/h] (default 3000)
            T_cool_K : float      Coolant temperature [K] (default 523.15)
        """
        T0 = inputs.get("T0_K", self._model.T_in)
        dur = inputs.get("duration_s", 600.0)
        dt = inputs.get("dt", 1.0)
        P = inputs.get("P_bar", None)
        GHSV = inputs.get("GHSV", None)
        T_cool = inputs.get("T_cool_K", None)

        result = self._model.simulate(
            T0=T0, duration_s=dur, dt=dt, P=P, GHSV=GHSV, T_cool=T_cool,
        )
        return result

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "T0_K": {"unit": "K", "range": [473.15, 873.15]},
                "duration_s": {"unit": "s"},
                "dt": {"unit": "s"},
                "P_bar": {"unit": "bar", "range": [1.0, 30.0]},
                "GHSV": {"unit": "1/h", "range": [1000, 10000]},
                "T_cool_K": {"unit": "K", "range": [473.15, 673.15]},
            },
            "outputs": {
                "t": "s",
                "T": "K",
                "X_CO2": "- (CO2 conversion)",
                "y_CH4_dry": "- (dry CH4 mole fraction)",
                "C_CO2": "mol/m3",
                "C_H2": "mol/m3",
                "C_CH4": "mol/m3",
                "C_H2O": "mol/m3",
                "T_max": "K",
                "thermal_runaway": "bool",
                "X_eq_final": "- (equilibrium conversion at final T)",
            },
            "source": self._raw.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    r = m.predict({"T0_K": 523.15, "duration_s": 300.0, "dt": 5.0})
    print(f"Final T: {r['T'][-1]:.1f} K")
    print(f"Final CO2 conversion: {r['X_CO2'][-1]:.4f}")
    print(f"Final CH4 dry fraction: {r['y_CH4_dry'][-1]:.4f}")
    print(f"Thermal runaway: {r['thermal_runaway']}")
