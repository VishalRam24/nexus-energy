"""
EC087 -- Biomass Boiler -- F2a Combustion + Thermal-Mass ODE
Standardised predict() / get_info() interface (mirrors EC001 F2a).
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import BiomassBoilerF2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for the EC087 F2a physics-lumped biomass boiler."""

    component_id = "EC087"
    component_name = "Biomass Boiler"
    fidelity = "F2a -- Combustion + Flue-Gas Energy Balance + Lumped Thermal-Mass ODE"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            # allow overriding individual unit values (e.g. moisture_content)
            for k, v in params.items():
                if k in self._raw["unit"]:
                    self._raw["unit"][k]["value"] = v
        self._model = BiomassBoilerF2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        Run dynamic boiler simulation.

        inputs:
            PLR            : float or callable PLR(t), part-load ratio in [0,1]
            T_water_init_K : float  (initial block temperature, default 333.15)
            T_return_K     : float  (return-water temperature, default 323.15)
            dt             : float  (output step, default 2.0 s)
            duration_s     : float  (default 1200 s)
        """
        PLR = inputs.get("PLR", 1.0)
        T0 = inputs.get("T_water_init_K", 333.15)
        T_ret = inputs.get("T_return_K", 323.15)
        dt = inputs.get("dt", 2.0)
        dur = inputs.get("duration_s", 1200.0)
        return self._model.simulate(PLR, T0, T_ret, dt, dur)

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "PLR": {"unit": "-", "range": [0.0, 1.0]},
                "T_water_init_K": {"unit": "K", "range": [280, 370]},
                "T_return_K": {"unit": "K", "range": [280, 360]},
                "dt": {"unit": "s"},
                "duration_s": {"unit": "s"},
            },
            "outputs": {
                "t": "s",
                "T_water": "K",
                "PLR": "-",
                "m_fuel": "kg/s",
                "m_air": "kg/s",
                "m_flue": "kg/s",
                "T_flue_C": "degC",
                "Q_comb": "kW",
                "Q_stack": "kW",
                "Q_casing": "kW",
                "Q_useful": "kW",
                "efficiency": "-",
                "LHV_eff": "kJ/kg",
            },
            "source": self._raw.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    r = m.predict({"PLR": 1.0, "duration_s": 1800.0, "dt": 5.0})
    print(f"Final T_water: {r['T_water'][-1]-273.15:.2f} degC, "
          f"eta_ss: {r['efficiency'][-1]:.3f}, "
          f"Q_useful: {r['Q_useful'][-1]:.2f} kW, "
          f"LHV_eff: {r['LHV_eff']:.0f} kJ/kg")
