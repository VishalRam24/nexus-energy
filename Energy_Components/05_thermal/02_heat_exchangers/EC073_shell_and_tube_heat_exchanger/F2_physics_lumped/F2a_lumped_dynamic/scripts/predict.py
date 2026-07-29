"""
EC073 -- Shell-and-Tube Heat Exchanger -- F2a Lumped-Capacitance Transient
Standardised predict() / get_info() interface.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import ShellAndTubeHEX_F2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for the EC073 F2a lumped transient HX model."""

    component_id = "EC073"
    component_name = "Shell-and-Tube Heat Exchanger"
    fidelity = "F2a -- Lumped-Capacitance Transient (N control volumes + wall thermal mass)"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            self._raw["unit"].update(params)
        self._model = ShellAndTubeHEX_F2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        Run a transient simulation of the heat exchanger.

        inputs:
            T_h_in : float or callable(t)  hot inlet temperature [degC]
            T_c_in : float or callable(t)  cold inlet temperature [degC]
            m_dot_hot : float              hot mass flow [kg/s]
            m_dot_cold : float             cold mass flow [kg/s]
            duration_s : float             horizon [s]  (default 300)
            dt : float                     output interval [s] (default 1.0)
            T_init : float or None         uniform initial temp [degC]

        Returns time-series outlet temperatures, heat duty, and the
        steady-state epsilon-NTU reference duty for comparison.
        """
        T_h_in = inputs.get("T_h_in", 90.0)
        T_c_in = inputs.get("T_c_in", 20.0)
        m_h = inputs.get("m_dot_hot", 2.0)
        m_c = inputs.get("m_dot_cold", 2.0)
        dur = inputs.get("duration_s", 300.0)
        dt = inputs.get("dt", 1.0)
        T_init = inputs.get("T_init", None)

        res = self._model.simulate(T_h_in, T_c_in, m_h, m_c,
                                   duration_s=dur, dt=dt, T_init=T_init)

        # Steady-state reference (uses inlet values at t=0 if callables)
        T_h0 = T_h_in(0.0) if callable(T_h_in) else T_h_in
        T_c0 = T_c_in(0.0) if callable(T_c_in) else T_c_in
        ss = self._model.steady_state_duty(T_h0, T_c0, m_h, m_c)
        res["steady_state_reference"] = ss
        return res

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "T_h_in": {"unit": "degC", "range": [30, 200]},
                "T_c_in": {"unit": "degC", "range": [5, 80]},
                "m_dot_hot": {"unit": "kg/s", "range": [0.1, 50]},
                "m_dot_cold": {"unit": "kg/s", "range": [0.1, 50]},
                "duration_s": {"unit": "s"},
                "dt": {"unit": "s"},
                "T_init": {"unit": "degC"},
            },
            "outputs": {
                "t": "s",
                "T_h_out": "degC (array)",
                "T_c_out": "degC (array)",
                "Q_kw": "kW (array)",
                "T_h_profile": "degC node temperatures (N x n_t)",
                "T_w_profile": "degC wall temperatures (N x n_t)",
                "T_c_profile": "degC node temperatures (N x n_t)",
                "steady_state_reference": "dict (epsilon-NTU limit)",
            },
            "source": self._raw["unit"].get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    r = m.predict({"T_h_in": 90.0, "T_c_in": 20.0,
                   "m_dot_hot": 2.0, "m_dot_cold": 2.0,
                   "duration_s": 400.0, "dt": 5.0})
    ss = r["steady_state_reference"]
    print(f"\nTransient final:  T_h_out={r['T_h_out'][-1]:.2f} degC, "
          f"T_c_out={r['T_c_out'][-1]:.2f} degC, Q={r['Q_kw'][-1]:.2f} kW")
    print(f"eps-NTU steady:   T_h_out={ss['T_h_out']:.2f} degC, "
          f"T_c_out={ss['T_c_out']:.2f} degC, Q={ss['Q_kw']:.2f} kW "
          f"(eps={ss['effectiveness']:.3f}, NTU={ss['ntu']:.3f})")
