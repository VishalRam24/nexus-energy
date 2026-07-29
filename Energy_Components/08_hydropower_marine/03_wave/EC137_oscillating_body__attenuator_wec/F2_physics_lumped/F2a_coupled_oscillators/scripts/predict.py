"""
EC137 -- Oscillating Body / Attenuator WEC (Pelamis-type) -- F2a
Standardised predict() / get_info() interface.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import AttenuatorWEC_F2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for the EC137 F2a coupled-oscillator attenuator model."""

    component_id = "EC137"
    component_name = "Oscillating Body / Attenuator WEC (Pelamis-type)"
    fidelity = "F2a -- Coupled Hinged-Raft Oscillators with Hydraulic PTO"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            self._raw["unit"].update(params)
        self._model = AttenuatorWEC_F2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        Run a time-domain attenuator simulation for a given sea state.

        inputs:
            H_s : float        significant / regular wave height [m]
            T_e : float        energy / wave period [s]
            B_pto : float      PTO damping per joint [N.m.s/rad] (optional)
            K_pto : float      PTO reactive stiffness per joint [N.m/rad] (optional)
            dt : float         output time step [s] (default 0.1)
            duration_s : float total simulated time [s] (default 120)
            optimize_pto : bool  if True, scan B_pto for max power (default False)

        returns: dict with mean power, capture width, time-series, etc.
        """
        H_s = inputs.get("H_s", 2.0)
        T_e = inputs.get("T_e", 8.0)
        B_pto = inputs.get("B_pto", None)
        K_pto = inputs.get("K_pto", None)
        dt = inputs.get("dt", 0.1)
        dur = inputs.get("duration_s", 120.0)
        optimize = inputs.get("optimize_pto", False)

        if optimize:
            opt = self._model.optimal_B_pto(H_s, T_e, dt=dt, duration_s=dur)
            B_pto = opt["B_opt"]

        result = self._model.simulate(H_s, T_e, B_pto=B_pto, K_pto=K_pto,
                                       dt=dt, duration_s=dur)
        if optimize:
            result["B_opt"] = opt["B_opt"]
            result["B_opt_analytic"] = opt["B_opt_analytic"]
        return result

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "H_s": {"unit": "m", "range": [0.25, 8.0]},
                "T_e": {"unit": "s", "range": [5.0, 20.0]},
                "B_pto": {"unit": "N.m.s/rad", "range": [1e6, 5e8]},
                "K_pto": {"unit": "N.m/rad"},
                "dt": {"unit": "s"},
                "duration_s": {"unit": "s"},
                "optimize_pto": {"unit": "bool"},
            },
            "outputs": {
                "mean_power_elec_W": "W",
                "mean_power_mech_W": "W",
                "capture_width_m": "m",
                "capture_width_ratio": "-",
                "wave_power_per_m_W": "W/m",
                "power_total_elec": "W (time series)",
                "theta": "rad (n_joint x N relative hinge angles)",
                "energy_residual": "- (excitation vs dissipation balance)",
            },
            "device": {
                "n_segments": self._model.n_seg,
                "n_joints": self._model.n_joint,
                "device_length_m": self._model.device_length,
                "device_width_m": self._model.device_width,
            },
            "source": self._raw.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    r = m.predict({"H_s": 3.0, "T_e": 9.0, "duration_s": 80.0, "dt": 0.1})
    print(f"\nSea state H_s=3.0 m, T_e=9.0 s:")
    print(f"  Mean electrical power : {r['mean_power_elec_W']/1e3:.1f} kW")
    print(f"  Wave resource         : {r['wave_power_per_m_W']/1e3:.1f} kW/m")
    print(f"  Capture width         : {r['capture_width_m']:.2f} m")
    print(f"  Capture width ratio   : {r['capture_width_ratio']:.3f}")
    print(f"  Energy residual       : {r['energy_residual']*100:.2f} %")
