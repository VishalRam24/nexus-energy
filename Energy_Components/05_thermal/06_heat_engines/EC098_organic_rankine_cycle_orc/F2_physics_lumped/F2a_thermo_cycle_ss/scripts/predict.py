"""
EC098 -- Organic Rankine Cycle (ORC) -- F2a Thermo Cycle Steady-State
Standardised predict() / get_info() interface.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import ORC_F2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for ORC F2a thermodynamic cycle model."""

    component_id = "EC098"
    component_name = "Organic Rankine Cycle (ORC)"
    fidelity = "F2a -- Thermodynamic Cycle Steady-State with Part-Load"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            self._raw["unit"].update(params)
        self._model = ORC_F2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        Run ORC cycle calculation or dynamic simulation.

        inputs:
            P_evap : float  (evaporator pressure [Pa], default from params)
            P_cond : float  (condenser pressure [Pa], default from params)
            load_fraction : float  (0.1 to 1.0, default 1.0)
            superheat : float  (K, default from params)
            mode : str  ('steady' or 'dynamic', default 'steady')
            T_ambient_K : float  (for dynamic mode, default 293.15)
            dt : float  (for dynamic mode, default 1.0)
            duration_s : float  (for dynamic mode, default 3600.0)
        """
        mode = inputs.get("mode", "steady")

        if mode == "steady":
            return self._model.compute_cycle(
                P_evap=inputs.get("P_evap"),
                P_cond=inputs.get("P_cond"),
                superheat=inputs.get("superheat"),
                load_fraction=inputs.get("load_fraction", 1.0),
            )
        else:
            return self._model.simulate(
                load_profile=inputs.get("load_fraction", 1.0),
                T_ambient_K=inputs.get("T_ambient_K", 293.15),
                P_evap=inputs.get("P_evap"),
                P_cond=inputs.get("P_cond"),
                dt=inputs.get("dt", 1.0),
                duration_s=inputs.get("duration_s", 3600.0),
            )

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "P_evap": {"unit": "Pa", "range": [500000, 3000000]},
                "P_cond": {"unit": "Pa", "range": [100000, 500000]},
                "load_fraction": {"unit": "-", "range": [0.1, 1.0]},
                "superheat": {"unit": "K", "range": [0, 30]},
                "mode": {"options": ["steady", "dynamic"]},
            },
            "outputs": {
                "W_net": "W",
                "eta_thermal": "-",
                "Q_in": "W",
                "Q_out": "W",
                "m_dot": "kg/s",
                "state_points": "dict of T/P/h/s arrays",
            },
            "source": self._raw.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    r = m.predict({"load_fraction": 1.0})
    print(f"W_net: {r['W_net']/1000:.1f} kW, eta: {r['eta_thermal']:.4f}, "
          f"m_dot: {r['m_dot']:.3f} kg/s")
