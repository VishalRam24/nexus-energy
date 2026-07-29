"""
EC061 — Unglazed Solar Collector (Pool Heating) — F2a Physics-Lumped
Standardised predict() / get_info() interface (mirrors EC001 F2a).
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import UnglazedCollectorF2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for the EC061 F2a physics-lumped unglazed collector."""

    component_id = "EC061"
    component_name = "Unglazed Solar Collector (Pool Heating)"
    fidelity = "F2a — Physics-Lumped Dynamic Energy Balance (wind-dependent losses)"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            self._raw["unit"].update(params)
        self._model = UnglazedCollectorF2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        Run the dynamic lumped energy-balance simulation.

        inputs:
            G          : float|callable  irradiance on aperture [W/m2]
            Ta         : float|callable  ambient air temperature [degC] (default 20)
            Tsky       : float|callable  sky temperature [degC] (default Ta-6)
            u_wind     : float|callable  wind speed [m/s] (default 1.0)
            Tf_in      : float|callable  pool/fluid inlet temperature [degC] (default Ta)
            Tp0        : float           initial plate temperature [degC] (default Ta)
            dt         : float           output step [s] (default 60)
            duration_s : float           horizon [s] (default 3600)
        """
        G = inputs.get("G", 800.0)
        Ta = inputs.get("Ta", 20.0)
        Tsky = inputs.get("Tsky", None)
        u_wind = inputs.get("u_wind", 1.0)
        Tf_in = inputs.get("Tf_in", None)
        Tp0 = inputs.get("Tp0", None)
        dt = inputs.get("dt", 60.0)
        dur = inputs.get("duration_s", 3600.0)

        return self._model.simulate(G, Ta, Tsky, u_wind, Tf_in, Tp0, dt, dur)

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "G": {"unit": "W/m2", "range": [0, 1100]},
                "Ta": {"unit": "degC", "range": [0, 40]},
                "Tsky": {"unit": "degC", "range": [-20, 40]},
                "u_wind": {"unit": "m/s", "range": [0, 12]},
                "Tf_in": {"unit": "degC", "range": [10, 40]},
                "Tp0": {"unit": "degC"},
                "dt": {"unit": "s"},
                "duration_s": {"unit": "s"},
            },
            "outputs": {
                "t": "s",
                "T_plate": "degC",
                "q_use": "W/m2",
                "Q_use_W": "W",
                "eta": "-",
                "U_L": "W/m2K",
                "eta0_eff": "-",
            },
            "source": self._raw.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    r = m.predict({"G": 800.0, "Ta": 22.0, "u_wind": 1.0, "Tf_in": 24.0,
                   "dt": 30.0, "duration_s": 1800.0})
    print(f"Final plate T: {r['T_plate'][-1]:.2f} degC, "
          f"Q_use: {r['Q_use_W'][-1]:.1f} W, eta: {r['eta'][-1]:.3f}")
