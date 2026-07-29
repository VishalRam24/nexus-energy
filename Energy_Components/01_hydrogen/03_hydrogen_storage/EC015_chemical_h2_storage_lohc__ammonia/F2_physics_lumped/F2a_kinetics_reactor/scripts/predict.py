"""
EC015 -- Chemical H2 Storage (LOHC / Ammonia) -- F2a Lumped Kinetics Reactor
Standardised predict() / get_info() interface.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import ChemicalH2StorageF2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for EC015 F2a lumped kinetics reactor."""

    component_id = "EC015"
    component_name = "Chemical H2 Storage (LOHC / Ammonia)"
    fidelity = "F2a -- Lumped Kinetics + Energy-Balance Reactor (solve_ivp)"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            self._raw.update(params)
        self._model = ChemicalH2StorageF2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        Run dynamic batch-reactor dehydrogenation / cracking simulation.

        inputs:
            mode : 'lohc' or 'ammonia'   (default 'lohc')
            T0 : float        initial reactor temperature [K]
            T_set : float     heater setpoint [K]
            n_carrier : float initial moles of hydrogenated carrier [mol]
            dt : float        output step [s] (default 10)
            duration_s : float total sim time [s] (default 3600)
            X0 : float        initial conversion (default 0)
            hA : float        heater conductance [W/K]
        """
        mode = inputs.get("mode", "lohc")
        return self._model.simulate(
            mode=mode,
            T0=inputs.get("T0"),
            T_set=inputs.get("T_set"),
            n_carrier=inputs.get("n_carrier"),
            dt=inputs.get("dt", 10.0),
            duration_s=inputs.get("duration_s", 3600.0),
            X0=inputs.get("X0", 0.0),
            hA=inputs.get("hA"),
        )

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "mode": {"values": ["lohc", "ammonia"]},
                "T0": {"unit": "K", "range": [423, 873]},
                "T_set": {"unit": "K", "range": [423, 873]},
                "n_carrier": {"unit": "mol", "range": [0.1, 1e6]},
                "dt": {"unit": "s"},
                "duration_s": {"unit": "s"},
                "X0": {"unit": "-", "range": [0, 1]},
                "hA": {"unit": "W/K"},
            },
            "outputs": {
                "t": "s",
                "conversion": "-",
                "h2_rate_mol_s": "mol/s",
                "h2_rate_kg_s": "kg/s",
                "h2_released_kg": "kg",
                "temperature": "K",
                "q_heat_W": "W",
                "q_rxn_W": "W",
                "specific_energy_MJ_per_kg": "MJ/kg_H2",
                "energy_penalty_frac": "-",
            },
            "source": self._raw.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    r = m.predict({"mode": "lohc", "duration_s": 3600.0, "dt": 60.0})
    print(f"[LOHC] final X={r['conversion'][-1]:.4f}, "
          f"H2 released={r['h2_released_kg'][-1]:.4f} kg, "
          f"specific energy={r['specific_energy_MJ_per_kg']:.2f} MJ/kg, "
          f"penalty={r['energy_penalty_frac']*100:.1f}% of LHV")
    r2 = m.predict({"mode": "ammonia", "duration_s": 3600.0, "dt": 60.0})
    print(f"[NH3]  final X={r2['conversion'][-1]:.4f}, "
          f"H2 released={r2['h2_released_kg'][-1]:.4f} kg, "
          f"specific energy={r2['specific_energy_MJ_per_kg']:.2f} MJ/kg, "
          f"penalty={r2['energy_penalty_frac']*100:.1f}% of LHV")
