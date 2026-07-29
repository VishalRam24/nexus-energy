"""
EC138 -- Ocean Thermal Energy Conversion (OTEC) -- F2a Physics-Lumped Thermal Cycle
Standardised predict() / get_info() interface.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import OTEC_F2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for OTEC F2a closed-cycle ammonia Rankine model."""

    component_id = "EC138"
    component_name = "Ocean Thermal Energy Conversion (OTEC)"
    fidelity = "F2a -- Physics-Lumped Closed-Cycle Ammonia Rankine with HX Thermal ODE"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            self._raw["unit"].update(params)
        self._model = OTEC_F2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        Run the lumped-HX transient simulation and return the final operating point
        plus the full time series.

        inputs:
            T_warm_in_c : float  warm surface seawater inlet [degC] (default 26)
            T_cold_in_c : float  cold deep seawater inlet [degC]    (default 5)
            mdot_warm_kg_s : float  warm seawater flow [kg/s]       (default 3000)
            mdot_cold_kg_s : float  cold seawater flow [kg/s]       (default 2200)
            dt : float           output step [s]                    (default 10)
            duration_s : float   sim horizon [s]                    (default 3600)
        """
        Tw = inputs.get("T_warm_in_c", self._model.T_warm_in)
        Tc = inputs.get("T_cold_in_c", self._model.T_cold_in)
        mw = inputs.get("mdot_warm_kg_s", self._model.mdot_warm)
        mc = inputs.get("mdot_cold_kg_s", self._model.mdot_cold)
        dt = inputs.get("dt", 10.0)
        dur = inputs.get("duration_s", 3600.0)

        ts = self._model.simulate(Tw, Tc, mw, mc, dt, dur)
        ss = self._model.steady_state(Tw, Tc, mw, mc)

        return {
            "t": ts["t"],
            "T_evap_c": ts["T_evap_c"],
            "T_cond_c": ts["T_cond_c"],
            "eta_carnot": ts["eta_carnot"],
            "eta_cycle": ts["eta_cycle"],
            "eta_net": ts["eta_net"],
            "P_gross_kw": ts["P_gross_kw"],
            "P_parasitic_kw": ts["P_parasitic_kw"],
            "P_net_kw": ts["P_net_kw"],
            "Q_evap_kw": ts["Q_evap_kw"],
            # scalar final / steady-state summary
            "eta_carnot_final": float(ts["eta_carnot"][-1]),
            "eta_net_final": float(ts["eta_net"][-1]),
            "P_gross_final_kw": float(ts["P_gross_kw"][-1]),
            "P_net_final_kw": float(ts["P_net_kw"][-1]),
            "steady_state": ss,
        }

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "T_warm_in_c": {"unit": "degC", "range": [18.0, 32.0]},
                "T_cold_in_c": {"unit": "degC", "range": [2.0, 12.0]},
                "mdot_warm_kg_s": {"unit": "kg/s", "range": [500.0, 6000.0]},
                "mdot_cold_kg_s": {"unit": "kg/s", "range": [400.0, 5000.0]},
                "dt": {"unit": "s"},
                "duration_s": {"unit": "s"},
            },
            "outputs": {
                "t": "s",
                "T_evap_c": "degC",
                "T_cond_c": "degC",
                "eta_carnot": "-",
                "eta_cycle": "-",
                "eta_net": "-",
                "P_gross_kw": "kW",
                "P_parasitic_kw": "kW",
                "P_net_kw": "kW",
                "Q_evap_kw": "kW",
            },
            "source": self._raw.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    r = m.predict({"T_warm_in_c": 26.0, "T_cold_in_c": 5.0,
                   "duration_s": 3600.0, "dt": 30.0})
    print(f"\nCarnot eff      : {r['eta_carnot_final']*100:.2f} %")
    print(f"Net eff         : {r['eta_net_final']*100:.2f} %")
    print(f"Gross power     : {r['P_gross_final_kw']:.1f} kW")
    print(f"Net power       : {r['P_net_final_kw']:.1f} kW")
    ss = r["steady_state"]
    print(f"Parasitic       : {ss['P_parasitic_kw']:.1f} kW "
          f"(warm {ss['P_warm_pump_kw']:.1f}, cold {ss['P_cold_pump_kw']:.1f}, "
          f"wf {ss['P_wf_pump_kw']:.1f})")
