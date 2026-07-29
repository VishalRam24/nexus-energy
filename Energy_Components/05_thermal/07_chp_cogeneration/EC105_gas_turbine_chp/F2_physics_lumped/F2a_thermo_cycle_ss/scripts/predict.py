"""
EC105 -- Gas Turbine CHP -- F2a Physics-Lumped Thermo Cycle
Standardised predict() / get_info() interface (mirrors EC001 template).
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import GasTurbineCHP_F2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for EC105 Gas Turbine CHP F2a physics-lumped model."""

    component_id = "EC105"
    component_name = "Gas Turbine CHP"
    fidelity = "F2a -- Physics-Lumped Brayton Topping Cycle + HRSG Thermal ODE"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            self._raw["unit"].update(params)
        self._model = GasTurbineCHP_F2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        Run the steady CHP cycle solve and (optionally) the HRSG transient.

        inputs:
            part_load_ratio   : float (default 1.0)
            T_amb_K           : float (default param ambient)
            pressure_ratio    : float (default param rp)
            T_turbine_inlet_K : float (default param TIT)
            transient         : bool  (default True) -- also integrate HRSG ODE
            dt                : float (default 2.0)
            duration_s        : float (default 600.0)

        Returns the steady-state CHP performance plus, if transient, the HRSG
        warm-up time history.
        """
        plr = inputs.get("part_load_ratio", 1.0)
        T_amb = inputs.get("T_amb_K", None)
        rp = inputs.get("pressure_ratio", None)
        T3 = inputs.get("T_turbine_inlet_K", None)
        transient = inputs.get("transient", True)
        dt = inputs.get("dt", 2.0)
        dur = inputs.get("duration_s", 600.0)

        state = self._model.cycle_state(plr, T_amb, rp, T3)
        out = {
            "electrical_power_kw": state["electrical_power_w"] / 1e3,
            "thermal_power_kw": state["thermal_power_w"] / 1e3,
            "fuel_power_kw": state["fuel_power_w"] / 1e3,
            "eta_electrical": state["eta_electrical"],
            "eta_thermal": state["eta_thermal"],
            "eta_total": state["eta_total"],
            "eta_carnot": state["eta_carnot"],
            "heat_to_power_ratio": state["heat_to_power_ratio"],
            "T_exhaust_K": state["T4_K"],
            "T_compressor_out_K": state["T2_K"],
            "mdot_fuel_kgs": state["mdot_fuel_kgs"],
        }

        if transient:
            sim = self._model.simulate(plr, T_amb, rp, T3, dt=dt, duration_s=dur)
            out["t"] = sim["t"]
            out["T_hrsg_K"] = sim["T_hrsg_K"]
            out["thermal_power_transient_kw"] = sim["thermal_power_w"] / 1e3
            out["T_hrsg_steady_K"] = sim["T_hrsg_steady_K"]

        return out

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "part_load_ratio": {"unit": "-", "range": [0.4, 1.0]},
                "T_amb_K": {"unit": "K", "range": [248.15, 323.15]},
                "pressure_ratio": {"unit": "-", "range": [5.0, 30.0]},
                "T_turbine_inlet_K": {"unit": "K", "range": [1100.0, 1700.0]},
                "transient": {"unit": "bool"},
                "dt": {"unit": "s"},
                "duration_s": {"unit": "s"},
            },
            "outputs": {
                "electrical_power_kw": "kW_e",
                "thermal_power_kw": "kW_th",
                "fuel_power_kw": "kW (LHV)",
                "eta_electrical": "-",
                "eta_thermal": "-",
                "eta_total": "- (CHP fuel utilisation)",
                "eta_carnot": "- (power-cycle upper bound)",
                "heat_to_power_ratio": "-",
                "T_hrsg_K": "K (transient HRSG metal temperature)",
            },
            "source": self._raw.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    info = m.get_info()
    print(f"{info['component_id']} {info['component_name']} -- {info['fidelity']}")
    r = m.predict({"part_load_ratio": 1.0, "duration_s": 300.0, "dt": 5.0})
    print(f"  P_el      = {r['electrical_power_kw']:.1f} kW_e")
    print(f"  Q_th      = {r['thermal_power_kw']:.1f} kW_th")
    print(f"  eta_el    = {r['eta_electrical']*100:.1f} %")
    print(f"  eta_th    = {r['eta_thermal']*100:.1f} %")
    print(f"  eta_total = {r['eta_total']*100:.1f} %  (Carnot bound {r['eta_carnot']*100:.1f} %)")
    print(f"  HPR       = {r['heat_to_power_ratio']:.2f}")
    print(f"  HRSG warmed {r['T_hrsg_K'][0]:.1f} K -> {r['T_hrsg_K'][-1]:.1f} K "
          f"(steady {r['T_hrsg_steady_K']:.1f} K)")
