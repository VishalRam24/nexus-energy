"""
EC056 -- Linear Fresnel CSP -- F2a Physics-Lumped Receiver Thermal ODE
Standardised predict() / get_info() interface (mirrors EC001 template).
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import LinearFresnelF2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for the Linear Fresnel F2a lumped receiver ODE."""

    component_id = "EC056"
    component_name = "Linear Fresnel CSP"
    fidelity = "F2a -- Physics-Lumped Receiver Thermal-Balance ODE"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            self._raw["unit"].update(params)
        self._model = LinearFresnelF2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        Run the dynamic receiver-wall simulation.

        inputs:
            dni           : float or callable(t)  [W/m2]
            theta_L_deg   : float or callable(t)  longitudinal incidence angle
            theta_T_deg   : float or callable(t)  transversal incidence angle
            T_ambient_C   : float or callable(t)  [degC]
            T_htf_in_C    : float or callable(t)  receiver inlet temp [degC]
            T_wall0_C     : float (optional) initial wall temperature [degC]
            dt            : float (default 5.0)   output step [s]
            duration_s    : float (default 1800)  total duration [s]
        """
        dni = inputs.get("dni", 800.0)
        theta_L = inputs.get("theta_L_deg", 0.0)
        theta_T = inputs.get("theta_T_deg", 0.0)
        T_amb_C = inputs.get("T_ambient_C", 25.0)
        T_in_C = inputs.get("T_htf_in_C", 200.0)
        T_w0_C = inputs.get("T_wall0_C", None)
        dt = inputs.get("dt", 5.0)
        dur = inputs.get("duration_s", 1800.0)

        # degC -> K (handle scalar or callable)
        def to_K(x):
            if callable(x):
                return lambda t: x(t) + 273.15
            return x + 273.15

        T_amb_K = to_K(T_amb_C)
        T_in_K = to_K(T_in_C)
        T_w0_K = None if T_w0_C is None else T_w0_C + 273.15

        res = self._model.simulate(dni, theta_L, theta_T, T_amb_K, T_in_K,
                                   T_w0_K=T_w0_K, dt=dt, duration_s=dur)

        # convenience degC mirrors
        res["T_wall_C"] = res["T_wall_K"] - 273.15
        res["T_htf_out_C"] = res["T_htf_out_K"] - 273.15
        return res

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "dni": {"unit": "W/m2", "range": [0, 1200]},
                "theta_L_deg": {"unit": "deg", "range": [0, 80]},
                "theta_T_deg": {"unit": "deg", "range": [0, 60]},
                "T_ambient_C": {"unit": "degC", "range": [-10, 50]},
                "T_htf_in_C": {"unit": "degC", "range": [100, 400]},
                "T_wall0_C": {"unit": "degC"},
                "dt": {"unit": "s"},
                "duration_s": {"unit": "s"},
            },
            "outputs": {
                "t": "s",
                "T_wall_K": "K", "T_wall_C": "degC",
                "T_htf_out_K": "K", "T_htf_out_C": "degC",
                "q_abs_per_m": "W/m", "q_conv_per_m": "W/m",
                "q_rad_per_m": "W/m", "q_htf_per_m": "W/m",
                "Q_to_fluid_W": "W",
                "eta_thermal": "-", "eta_optical": "-",
                "P_electric_W": "W",
            },
            "source": self._raw.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    r = m.predict({"dni": 850.0, "theta_L_deg": 15.0, "theta_T_deg": 20.0,
                   "T_htf_in_C": 200.0, "duration_s": 1200.0, "dt": 10.0})
    print(f"Final wall T: {r['T_wall_C'][-1]:.2f} C, "
          f"HTF out T: {r['T_htf_out_C'][-1]:.2f} C, "
          f"eta_thermal: {r['eta_thermal'][-1]:.3f}, "
          f"P_elec: {r['P_electric_W'][-1]/1e3:.1f} kW")
