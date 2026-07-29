"""
EC211 -- Forward Osmosis (FO) -- F2a Physics-Lumped Osmotic Flux Model
Standardised predict() / get_info() interface (mirrors EC001 template).
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import ForwardOsmosisF2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for FO F2a osmotic-flux physics-lumped model."""

    component_id = "EC211"
    component_name = "Forward Osmosis (FO)"
    fidelity = "F2a -- Physics-Lumped Osmotic Flux with Concentration Polarization"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            self._raw["unit"].update(params)
        self._model = ForwardOsmosisF2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        Run lumped FO module simulation (draw dilution ODE).

        inputs:
            c_draw0_mol_m3 : float  (initial draw concentration, default param)
            c_feed_mol_m3  : float  (feed concentration, default param)
            V_draw0_m3     : float  (initial draw volume, default param)
            duration_s     : float  (default 3600)
            n_points       : int    (default 200)
        """
        c_draw0 = inputs.get("c_draw0_mol_m3", None)
        c_feed = inputs.get("c_feed_mol_m3", None)
        V_draw0 = inputs.get("V_draw0_m3", None)
        dur = inputs.get("duration_s", 3600.0)
        npts = int(inputs.get("n_points", 200))

        return self._model.simulate(
            duration_s=dur, n_points=npts,
            V_draw0=V_draw0, c_draw0=c_draw0, c_feed=c_feed,
        )

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "c_draw0_mol_m3": {"unit": "mol/m3", "range": [100, 5000]},
                "c_feed_mol_m3": {"unit": "mol/m3", "range": [0, 600]},
                "V_draw0_m3": {"unit": "m3"},
                "duration_s": {"unit": "s", "range": [1, 86400]},
                "n_points": {"unit": "-"},
            },
            "outputs": {
                "t": "s",
                "V_draw_m3": "m3",
                "c_draw_mol_m3": "mol/m3",
                "Jw_LMH": "L/(m2.h) water flux",
                "Js_gMH": "g/(m2.h) reverse salt flux",
                "pi_draw_bar": "bar",
                "pi_feed_bar": "bar",
                "permeate_m3": "m3 cumulative product water",
                "SEC_regen_kWh_m3": "kWh/m3 draw regeneration energy",
            },
            "source": self._raw.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    r = m.predict({"duration_s": 3600.0, "n_points": 50})
    print(f"Initial Jw: {r['Jw_LMH'][0]:.3f} LMH, "
          f"Final Jw: {r['Jw_LMH'][-1]:.3f} LMH")
    print(f"Draw conc: {r['c_draw_mol_m3'][0]:.1f} -> "
          f"{r['c_draw_mol_m3'][-1]:.1f} mol/m3")
    print(f"Permeate produced: {r['permeate_m3'][-1]*1000:.2f} L over 1 h, "
          f"SEC_regen={r['SEC_regen_kWh_m3']:.2f} kWh/m3")
