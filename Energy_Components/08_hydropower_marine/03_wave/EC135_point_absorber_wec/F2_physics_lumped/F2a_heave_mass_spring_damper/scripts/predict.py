"""
EC135 -- Point Absorber WEC -- F2a Heaving-Buoy Linear Hydrodynamic Model
Standardised predict() / get_info() interface.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import PointAbsorberF2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for the EC135 F2a heaving point-absorber WEC model."""

    component_id = "EC135"
    component_name = "Point Absorber Wave Energy Converter (WEC)"
    fidelity = "F2a -- Heaving-Buoy Linear Hydrodynamic Mass-Spring-Damper ODE"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            self._raw["unit"].update(params)
        self._model = PointAbsorberF2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        Run a regular-wave heave simulation and return the absorbed-power metrics.

        inputs:
            H_m         : regular wave height [m]            (default 1.5)
            T_s         : wave period [s]                    (default natural period)
            B_pto       : PTO damping [N.s/m]                (default from params;
                          pass "optimal" to use |Z_i| at T_s)
            C_pto       : PTO reactive stiffness [N/m]       (default 0)
            dt          : output time step [s]               (default 0.05)
            duration_s  : simulation length [s]              (default 60*T)
        """
        H = inputs.get("H_m", 1.5)
        T = inputs.get("T_s", self._model.natural_period())
        B_pto = inputs.get("B_pto", None)
        if B_pto == "optimal":
            B_pto = self._model.optimal_B_pto(T)
        C_pto = inputs.get("C_pto", None)
        dt = inputs.get("dt", 0.05)
        dur = inputs.get("duration_s", None)

        return self._model.simulate(H=H, T=T, B_pto=B_pto, C_pto=C_pto,
                                    dt=dt, duration_s=dur)

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "H_m": {"unit": "m", "range": [0.25, 8.0]},
                "T_s": {"unit": "s", "range": [4.0, 20.0]},
                "B_pto": {"unit": "N.s/m", "range": [0.0, 1.0e6]},
                "C_pto": {"unit": "N/m"},
                "dt": {"unit": "s"},
                "duration_s": {"unit": "s"},
            },
            "outputs": {
                "t": "s",
                "x": "m (heave displacement)",
                "x_dot": "m/s (heave velocity)",
                "P_pto_mean": "W (mean absorbed mechanical power)",
                "P_elec_mean": "W (mean electrical power)",
                "capture_width": "m",
                "capture_width_max": "m (theoretical bound)",
                "capture_width_ratio": "-",
                "T_natural": "s",
            },
            "source": self._raw.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    Tn = m._model.natural_period()
    r = m.predict({"H_m": 1.5, "T_s": Tn, "B_pto": "optimal", "duration_s": 30 * Tn})
    print(f"\nNatural period T_n = {Tn:.2f} s")
    print(f"At resonance, H=1.5 m, optimal B_pto = {r['B_pto']:.3e} N.s/m")
    print(f"  Heave amplitude   = {r['amplitude']:.3f} m")
    print(f"  Mean absorbed P   = {r['P_pto_mean']/1e3:.2f} kW")
    print(f"  Mean electrical P = {r['P_elec_mean']/1e3:.2f} kW")
    print(f"  Capture width     = {r['capture_width']:.2f} m "
          f"(max {r['capture_width_max']:.2f} m)")
