"""
EC147 -- Hydrothermal Liquefaction (HTL) -- F2a Physics-Lumped Kinetics
Standardised predict() / get_info() interface (mirrors EC001 template).
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import HTL_F2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for the HTL F2a lumped-kinetics + energy-balance model."""

    component_id = "EC147"
    component_name = "Hydrothermal Liquefaction (HTL)"
    fidelity = "F2a -- Lumped Reaction-Network Kinetics + Reactor Energy-Balance ODE"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            self._raw["unit"].update(params)
        self._model = HTL_F2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        Simulate an HTL batch run (composition + temperature trajectory).

        inputs:
            T_setpoint_C  : float  reactor setpoint, subcritical [C]   (default 350)
            residence_min : float  residence/reaction time [min]       (default 30)
            T0_C          : float  initial slurry temperature [C]      (default 200)
            biomass0      : float  initial biomass mass fraction        (default 1.0)
            P_MPa         : float  operating pressure [MPa]             (default 18.0)
            n_out         : int    number of output samples            (default 200)
        """
        return self._model.simulate(
            T_setpoint_C=inputs.get("T_setpoint_C", 350.0),
            residence_min=inputs.get("residence_min", 30.0),
            T0_C=inputs.get("T0_C", 200.0),
            biomass0=inputs.get("biomass0", 1.0),
            P_MPa=inputs.get("P_MPa", 18.0),
            n_out=inputs.get("n_out", 200),
        )

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "T_setpoint_C": {"unit": "C", "range": [250, 374]},
                "residence_min": {"unit": "min", "range": [0, 120]},
                "T0_C": {"unit": "C", "range": [20, 374]},
                "biomass0": {"unit": "mass_fraction", "range": [0, 1]},
                "P_MPa": {"unit": "MPa", "range": [10, 25]},
                "n_out": {"unit": "-"},
            },
            "outputs": {
                "t_min": "min",
                "biocrude_yield": "kg/kg_dry_feed (time series + final)",
                "aqueous_yield": "kg/kg_dry_feed",
                "gas_yield": "kg/kg_dry_feed",
                "solid_yield": "kg/kg_dry_feed",
                "conversion": "fraction of biomass reacted",
                "temperature_C": "C",
                "mass_total": "should equal biomass0 (mass conservation)",
                "subcritical": "bool (water kept subcritical liquid)",
            },
            "source": self._raw.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    r = m.predict({"T_setpoint_C": 350.0, "residence_min": 30.0})
    f = r["final"]
    print(
        f"\nT=350C, t=30min -> biocrude={f['biocrude_yield']:.3f}, "
        f"aqueous={f['aqueous_yield']:.3f}, gas={f['gas_yield']:.3f}, "
        f"solid={f['solid_yield']:.3f}, conversion={f['conversion']:.3f}"
    )
    print(
        f"Biocrude energy = {f['biocrude_energy_MJ_per_kg_feed']:.2f} MJ/kg feed; "
        f"subcritical water = {r['subcritical']}"
    )
