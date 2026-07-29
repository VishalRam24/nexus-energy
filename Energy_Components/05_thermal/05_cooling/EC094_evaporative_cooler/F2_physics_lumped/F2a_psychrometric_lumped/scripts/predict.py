"""
EC094 -- Evaporative Cooler -- F2a Psychrometric Heat-Mass Transfer (Physics-Lumped)
Standardised predict() / get_info() interface (mirrors EC001 template).
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import EvaporativeCooler_F2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for EC094 evaporative cooler F2a psychrometric model."""

    component_id = "EC094"
    component_name = "Evaporative Cooler"
    fidelity = "F2a -- Psychrometric Heat-Mass Transfer with Lumped Transient ODE"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            self._raw["unit"].update(params)
        self._model = EvaporativeCooler_F2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        Run the lumped transient evaporative-cooling simulation.

        inputs:
            T_db_C : float (or callable for time-varying inlet dry-bulb) [degC]
            RH : float (relative humidity, 0-1)
            m_dot_air_kg_s : float (default 1.0)
            T_pad0_C : float (initial pad temperature, default = inlet T_db)
            dt : float (default 1.0)
            duration_s : float (default 120.0)
        """
        T_db = inputs.get("T_db_C", 35.0)
        RH = inputs.get("RH", 0.30)
        m_dot = inputs.get("m_dot_air_kg_s", 1.0)
        T_pad0 = inputs.get("T_pad0_C", None)
        dt = inputs.get("dt", 1.0)
        dur = inputs.get("duration_s", 120.0)

        return self._model.simulate(T_db, RH, m_dot, T_pad0, dt, dur)

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "T_db_C": {"unit": "degC", "range": [10, 50]},
                "RH": {"unit": "-", "range": [0.0, 1.0]},
                "m_dot_air_kg_s": {"unit": "kg/s", "range": [0.05, 10.0]},
                "T_pad0_C": {"unit": "degC"},
                "dt": {"unit": "s"},
                "duration_s": {"unit": "s"},
            },
            "outputs": {
                "t": "s",
                "T_db": "degC",
                "T_wb": "degC",
                "T_pad": "degC",
                "T_out": "degC",
                "Q_sens_W": "W",
                "COP": "-",
                "steady_state": "dict (T_out, w_in, w_out, m_dot_water, energy_residual, ...)",
            },
            "source": self._raw.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    r = m.predict({"T_db_C": 38.0, "RH": 0.20, "duration_s": 120.0, "dt": 2.0})
    ss = r["steady_state"]
    print(
        f"Inlet 38C/20%RH -> T_wb={ss['T_wb']:.2f}C, "
        f"T_out_final={r['T_out'][-1]:.2f}C, eps={ss['eps_sat']:.3f}, "
        f"water={ss['m_dot_water_kg_s']*1000:.3f} g/s, "
        f"energy_residual={ss['energy_residual']*100:.2f}%"
    )
