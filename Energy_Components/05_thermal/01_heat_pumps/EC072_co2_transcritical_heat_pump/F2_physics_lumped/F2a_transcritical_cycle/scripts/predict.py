"""
EC072 -- CO2 Transcritical Heat Pump (R744) -- F2a Transcritical Cycle
Standardised predict() / get_info() interface (mirrors EC001 F2a template).
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import CO2TranscriticalHPF2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for the EC072 F2a transcritical CO2 cycle model."""

    component_id = "EC072"
    component_name = "CO2 Transcritical Heat Pump (R744)"
    fidelity = "F2a -- Transcritical Thermodynamic Cycle with Gas-Cooler Glide + Lumped Water ODE"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            self._raw["unit"].update(params)
        self._model = CO2TranscriticalHPF2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        Run the transient charging simulation of the heated-water side.

        inputs:
            T_source_c       : float  source (evaporator) air/brine temp [degC] (default 0)
            T_water_in_c     : float  initial water temperature [degC] (default 15)
            T_water_target_c : float  target water-outlet temperature [degC] (default 65)
            P_high_bar       : float or None  high-side pressure [bar]; None -> optimum
            Q_load_kW        : float  external heat draw [kW] (default 0)
            duration_s       : float  simulation horizon [s] (default 1800)

        returns: dict of time series (t, T_water, cop, Q_heat_kW, P_elec_kW) +
                 scalar summary (cop_design, P_high_bar, etc.).
        """
        T_source = inputs.get("T_source_c", 0.0)
        T_water_in = inputs.get("T_water_in_c", 15.0)
        T_target = inputs.get("T_water_target_c", 65.0)
        P_high = inputs.get("P_high_bar", None)
        Q_load = inputs.get("Q_load_kW", 0.0)
        dur = inputs.get("duration_s", 1800.0)

        res = self._model.simulate(T_source, T_water_in, T_target,
                                   P_high_bar=P_high, Q_load_kW=Q_load,
                                   duration_s=dur)
        # Design-point cycle (at initial water inlet) for a scalar summary.
        st = self._model.cycle_states(
            T_source, T_water_in, res["P_high_bar"])
        res["cop_design"] = st["cop"]
        res["T_discharge_c"] = st["T2"] - 273.15
        res["T_gc_out_c"] = st["T3"] - 273.15
        res["q_gc_kJ_kg"] = st["q_gc"]
        return res

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "T_source_c": {"unit": "degC", "range": [-20, 25]},
                "T_water_in_c": {"unit": "degC", "range": [5, 50]},
                "T_water_target_c": {"unit": "degC", "range": [40, 90]},
                "P_high_bar": {"unit": "bar", "range": [74, 130], "note": "None -> optimum"},
                "Q_load_kW": {"unit": "kW"},
                "duration_s": {"unit": "s"},
            },
            "outputs": {
                "t": "s",
                "T_water": "degC",
                "cop": "-",
                "Q_heat_kW": "kW",
                "P_elec_kW": "kW",
                "cop_design": "-",
                "P_high_bar": "bar",
                "T_discharge_c": "degC",
                "T_gc_out_c": "degC",
            },
            "source": self._raw.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    r = m.predict({"T_source_c": 5.0, "T_water_in_c": 15.0,
                   "T_water_target_c": 65.0, "duration_s": 1200.0})
    print(f"Design COP: {r['cop_design']:.3f}  "
          f"P_high(opt): {r['P_high_bar']:.1f} bar  "
          f"T_discharge: {r['T_discharge_c']:.1f} C  "
          f"T_gc_out: {r['T_gc_out_c']:.1f} C")
    print(f"Water: {r['T_water'][0]:.1f} -> {r['T_water'][-1]:.1f} C in "
          f"{r['t'][-1]:.0f} s")
