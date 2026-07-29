"""
EC070 -- Water-Source Heat Pump -- F2a Vapor-Compression Cycle
Standardised predict() / get_info() interface (mirrors EC001 template).
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import WaterSourceHP_F2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for the WSHP F2a vapor-compression cycle model."""

    component_id = "EC070"
    component_name = "Water-Source Heat Pump"
    fidelity = "F2a -- Vapor-Compression Cycle (SS thermo) + Lumped Water-Loop Thermal ODE"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            self._raw["unit"].update(params)
        self._model = WaterSourceHP_F2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        Two modes:

        mode="cycle" (default) -- steady-state cycle at given source/sink temps:
            T_source_c : float [degC]  source-water temperature
            T_sink_c   : float [degC]  sink/load-water temperature
          returns the full cycle dict (COP, capacities, refrigerant states).

        mode="transient" -- lumped water-loop warm-up ODE:
            T_source_c   : float [degC]
            T_load0_c    : float [degC]  initial load temperature
            Q_demand_W   : float [W]     load heat draw
            duration_s   : float [s]
            dt           : float [s]
            T_setpoint_c : float or None
          returns time-series dict (t, T_load, cop, Q_cond, W_comp, ...).
        """
        mode = inputs.get("mode", "cycle")
        if mode == "transient":
            return self._model.simulate(
                inputs.get("T_source_c", 12.0),
                inputs.get("T_load0_c", 30.0),
                inputs.get("Q_demand_W", 20000.0),
                inputs.get("duration_s", 1800.0),
                inputs.get("dt", 30.0),
                inputs.get("T_setpoint_c", None),
            )
        return self._model.cycle(
            inputs.get("T_source_c", 12.0),
            inputs.get("T_sink_c", 45.0),
        )

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "mode": {"unit": "-", "values": ["cycle", "transient"]},
                "T_source_c": {"unit": "degC", "range": [5, 30]},
                "T_sink_c": {"unit": "degC", "range": [25, 65]},
                "T_load0_c": {"unit": "degC", "range": [5, 65]},
                "Q_demand_W": {"unit": "W", "range": [0, 60000]},
                "duration_s": {"unit": "s"},
                "dt": {"unit": "s"},
                "T_setpoint_c": {"unit": "degC"},
            },
            "outputs": {
                "cop_heat": "-",
                "cop_cool": "-",
                "cop_carnot": "-",
                "Q_cond": "W (heating)",
                "Q_evap": "W (cooling)",
                "W_comp": "W",
                "m_dot": "kg/s",
                "T_load": "degC time-series (transient)",
            },
            "source": self._raw.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    c = m.predict({"mode": "cycle", "T_source_c": 12.0, "T_sink_c": 45.0})
    print(
        f"\n[cycle] COP_h={c['cop_heat']:.3f}  COP_Carnot={c['cop_carnot']:.3f}  "
        f"Q_cond={c['Q_cond']/1000:.2f} kW  W_comp={c['W_comp']/1000:.2f} kW  "
        f"m_dot={c['m_dot']:.4f} kg/s  PR={c['pressure_ratio']:.2f}"
    )
    tr = m.predict({"mode": "transient", "T_source_c": 12.0, "T_load0_c": 30.0,
                    "Q_demand_W": 20000.0, "duration_s": 1800.0, "dt": 60.0})
    print(
        f"[transient] T_load {tr['T_load'][0]:.2f} -> {tr['T_load'][-1]:.2f} degC "
        f"over {tr['t'][-1]:.0f}s,  final COP={tr['cop'][-1]:.3f}"
    )
