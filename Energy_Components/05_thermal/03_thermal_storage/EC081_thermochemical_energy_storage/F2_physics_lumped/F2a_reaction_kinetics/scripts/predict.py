"""
EC081 -- Thermochemical Energy Storage (CaO/Ca(OH)2) -- F2a Reaction Kinetics
Standardised predict() / get_info() interface (mirrors EC001 template).
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import ThermochemicalStorageF2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for EC081 F2a thermochemical kinetics model."""

    component_id = "EC081"
    component_name = "Thermochemical Energy Storage (CaO/Ca(OH)2)"
    fidelity = "F2a -- Reaction Kinetics + Lumped Reactor Energy Balance ODE"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            self._raw["unit"].update(params)
        self._model = ThermochemicalStorageF2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        Run a dynamic thermochemical reactor simulation.

        inputs:
            mode : str  'charge' | 'discharge' | 'hold'   (default 'discharge')
            X0 : float  initial conversion / SOC [0,1]    (default by mode)
            T0_K : float  initial bed temperature [K]      (default 873.15)
            T_source_K : float  source/HTF temperature [K] (default 673.15)
            P_h2o_Pa : float  water-vapour pressure [Pa]   (default from params)
            dt : float  output time step [s]               (default 10.0)
            duration_s : float  simulated time [s]         (default 3600.0)
        """
        mode = inputs.get("mode", "discharge")
        X0 = inputs.get("X0", None)
        T0 = inputs.get("T0_K", 663.15)
        T_source = inputs.get("T_source_K", 623.15)
        P_h2o = inputs.get("P_h2o_Pa", None)
        dt = inputs.get("dt", 10.0)
        dur = inputs.get("duration_s", 3600.0)

        return self._model.simulate(mode, X0, T0, T_source, P_h2o, dt, dur)

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "mode": {"unit": "-", "options": ["charge", "discharge", "hold"]},
                "X0": {"unit": "-", "range": [0.0, 1.0]},
                "T0_K": {"unit": "K", "range": [573.15, 873.15]},
                "T_source_K": {"unit": "K", "range": [573.15, 873.15]},
                "P_h2o_Pa": {"unit": "Pa", "range": [1000.0, 500000.0]},
                "dt": {"unit": "s"},
                "duration_s": {"unit": "s"},
            },
            "outputs": {
                "t": "s",
                "X": "- (conversion / SOC)",
                "SOC": "-",
                "temperature": "K",
                "T_eq": "K",
                "stored_energy_J": "J",
                "reaction_rate": "1/s",
                "Q_rxn_W": "W",
                "E_max_J": "J",
            },
            "derived": {
                "n_mol": self._model.n_mol,
                "E_max_J": self._model.E_max,
                "C_th_J_per_K": self._model.C_th,
                "T_eq_K": self._model.T_eq(),
            },
            "source": self._raw.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    info = m.get_info()
    print(f"{info['component_id']} {info['component_name']} -- {info['fidelity']}")
    print(f"E_max = {info['derived']['E_max_J']/3.6e6:.1f} kWh, "
          f"T_eq = {info['derived']['T_eq_K']-273.15:.1f} C")
    r = m.predict({"mode": "discharge", "duration_s": 3600.0, "dt": 60.0})
    print(f"Discharge: SOC {r['SOC'][0]:.3f} -> {r['SOC'][-1]:.3f}, "
          f"T {r['temperature'][0]-273.15:.1f} -> {r['temperature'][-1]-273.15:.1f} C, "
          f"peak Q_rxn {r['Q_rxn_W'].max()/1e3:.1f} kW")
