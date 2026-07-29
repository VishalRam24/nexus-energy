"""
EC145 -- Pyrolysis Reactor -- F2a Lumped Arrhenius Kinetics + Energy Balance
Standardised predict() / get_info() interface (mirrors EC001 template).
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import PyrolysisReactorF2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for EC145 Pyrolysis Reactor F2a kinetics model."""

    component_id = "EC145"
    component_name = "Pyrolysis Reactor"
    fidelity = "F2a -- Lumped Arrhenius Kinetics + Reactor Energy Balance"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            self._raw["unit"].update(params)
        self._model = PyrolysisReactorF2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        Run a dynamic pyrolysis simulation (kinetics + reactor energy balance).

        inputs:
            Q_ext_W      : float  external heating duty [W]   (default 2000)
            T0_K         : float  initial reactor temperature [K] (default 500.0)
            dt           : float  output step [s]             (default 0.5)
            duration_s   : float  total time [s]              (default 120.0)
            mode         : str    'dynamic' (default) or 'isothermal'
            T_isothermal : float  used when mode == 'isothermal' [K] (default 773.15)
        """
        mode = inputs.get("mode", "dynamic")
        if mode == "isothermal":
            T_iso = inputs.get("T_isothermal", 773.15)
            hold = inputs.get("duration_s", 600.0)
            return self._model.equilibrium_yields(T_iso, hold_s=hold)

        Q = inputs.get("Q_ext_W", 2000.0)
        T0 = inputs.get("T0_K", 500.0)
        dt = inputs.get("dt", 0.5)
        dur = inputs.get("duration_s", 120.0)
        return self._model.simulate(Q, T0, dt, dur)

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "Q_ext_W": {"unit": "W", "range": [0.0, 1.0e5]},
                "T0_K": {"unit": "K", "range": [290.0, 1300.0]},
                "dt": {"unit": "s", "range": [0.01, 5.0]},
                "duration_s": {"unit": "s", "range": [1.0, 7200.0]},
                "mode": {"options": ["dynamic", "isothermal"]},
                "T_isothermal": {"unit": "K", "range": [525.0, 1200.0]},
            },
            "outputs": {
                "t": "s",
                "temperature": "K",
                "y_biomass": "mass fraction",
                "y_gas": "mass fraction",
                "y_bio_oil": "mass fraction",
                "y_char": "mass fraction",
                "conversion": "-",
                "mass_residual": "- (|sum-1|, conservation check)",
                "energy_*_MJ_kg": "MJ per kg feed",
            },
            "source": self._raw.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    r = m.predict({"Q_ext_W": 3000.0, "T0_K": 600.0, "duration_s": 120.0, "dt": 1.0})
    print(
        f"Final T: {r['final_temperature']:.1f} K | conversion: {r['final_conversion']*100:.1f}% | "
        f"bio-oil: {r['bio_oil_yield']*100:.1f}%  char: {r['char_yield']*100:.1f}%  gas: {r['gas_yield']*100:.1f}%"
    )
    iso = m.predict({"mode": "isothermal", "T_isothermal": 773.15})
    print(
        f"Isothermal 500C equilibrium -> bio-oil {iso['bio_oil_yield']*100:.1f}%  "
        f"char {iso['char_yield']*100:.1f}%  gas {iso['gas_yield']*100:.1f}%"
    )
