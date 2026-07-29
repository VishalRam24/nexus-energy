"""
EC097 -- Rankine Cycle (Steam Turbine) -- F2a Physics-Lumped Thermo Cycle
Standardised predict() / get_info() interface.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import RankineCycleF2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for the Rankine cycle F2a thermodynamic model."""

    component_id = "EC097"
    component_name = "Rankine Cycle (Steam Turbine)"
    fidelity = "F2a -- Physics-Lumped Thermodynamic Cycle with Boiler-Drum ODE"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            self._raw["unit"].update(params)
        self._model = RankineCycleF2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        Run a steady-state cycle solve and (optionally) a boiler-drum transient.

        inputs:
            mode : "steady" (default) or "transient"
            -- steady --
            P_boiler_bar    : float (optional, default from params)
            T_superheat_C   : float (optional)
            P_condenser_bar : float (optional)
            reheat          : bool (default False)
            regeneration    : bool (default False)
            -- transient --
            Q_fuel_W   : float or callable(t) (default Q_fuel_design)
            T_drum0_K  : float (optional)
            dt         : float (default 10.0)
            duration_s : float (default 3000.0)

        Returns the model's steady-state dict, or a dict with the steady
        solution plus a "transient" sub-dict of time-series arrays.
        """
        mode = inputs.get("mode", "steady")

        steady = self._model.solve_cycle(
            P_boiler=inputs.get("P_boiler_bar"),
            T_superheat=inputs.get("T_superheat_C"),
            P_condenser=inputs.get("P_condenser_bar"),
            reheat=inputs.get("reheat", False),
            regeneration=inputs.get("regeneration", False),
        )

        if mode == "transient":
            Q_fuel = inputs.get("Q_fuel_W", self._model.Q_fuel_design)
            T_drum0 = inputs.get("T_drum0_K")
            dt = inputs.get("dt", 10.0)
            dur = inputs.get("duration_s", 3000.0)
            transient = self._model.simulate(
                Q_fuel, T_drum0_K=T_drum0, dt=dt, duration_s=dur)
            out = dict(steady)
            out["transient"] = transient
            return out

        return steady

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "mode": {"unit": "-", "options": ["steady", "transient"]},
                "P_boiler_bar": {"unit": "bar", "range": [30.0, 250.0]},
                "T_superheat_C": {"unit": "degC", "range": [250.0, 620.0]},
                "P_condenser_bar": {"unit": "bar", "range": [0.03, 1.0]},
                "reheat": {"unit": "-"},
                "regeneration": {"unit": "-"},
                "Q_fuel_W": {"unit": "W"},
                "T_drum0_K": {"unit": "K"},
                "dt": {"unit": "s", "range": [1.0, 60.0]},
                "duration_s": {"unit": "s", "range": [10.0, 36000.0]},
            },
            "outputs": {
                "w_turbine": "kJ/kg",
                "w_pump": "kJ/kg",
                "w_net": "kJ/kg",
                "q_boiler": "kJ/kg",
                "q_cond": "kJ/kg",
                "eta_thermal": "-",
                "eta_carnot": "-",
                "P_mech_W": "W",
                "P_elec_W": "W",
                "Q_in_W": "W",
                "heat_rate_kJ_per_kWh": "kJ/kWh",
                "x_turbine_exit": "-",
                "transient": "dict of time-series arrays (transient mode)",
            },
            "source": self._raw.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    r = m.predict({"reheat": True})
    print(f"eta_thermal = {r['eta_thermal']:.4f}  "
          f"(Carnot {r['eta_carnot']:.4f}), "
          f"P_elec = {r['P_elec_W']/1e6:.2f} MW, "
          f"turbine exit quality = {r['x_turbine_exit']:.3f}")
    rt = m.predict({"mode": "transient", "duration_s": 2000.0, "dt": 20.0})
    tr = rt["transient"]
    print(f"transient: T_drum {tr['T_drum'][0]-273.15:.1f} -> "
          f"{tr['T_drum'][-1]-273.15:.1f} degC over {tr['t'][-1]:.0f} s, "
          f"final eta {tr['eta_thermal'][-1]:.4f}")
