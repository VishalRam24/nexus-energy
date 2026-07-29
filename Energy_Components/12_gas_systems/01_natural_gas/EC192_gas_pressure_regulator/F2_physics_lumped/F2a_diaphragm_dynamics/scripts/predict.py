"""
EC192 -- Gas Pressure Regulator -- F2a Physics-Lumped Diaphragm/Valve Dynamics
Standardised predict() / get_info() interface (mirrors EC001 template).
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import GasPressureRegulatorF2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for the EC192 F2a regulator dynamics model."""

    component_id = "EC192"
    component_name = "Gas Pressure Regulator"
    fidelity = "F2a -- Physics-Lumped Diaphragm/Valve Dynamics + Downstream-Volume ODE"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            self._raw["unit"].update(params)
        self._model = GasPressureRegulatorF2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        Run a dynamic regulator simulation.

        inputs:
            P_up_bar         : float upstream pressure [bar]            (default 50)
            load_flow_m3_h   : float downstream load draw [std m^3/h]   (default 1000)
                               (converted to kg/s internally)
            T_up_K           : float upstream temperature [K]           (default 288.15)
            duration_s       : float simulation horizon [s]             (default 60)
            dt               : float output step [s]                    (default 0.05)
            P_d0_bar         : float initial downstream pressure [bar]  (default setpoint)
        """
        P_up = inputs.get("P_up_bar", 50.0)
        load_m3h = inputs.get("load_flow_m3_h", 10000.0)
        T_up = inputs.get("T_up_K", 288.15)
        dur = inputs.get("duration_s", 60.0)
        dt = inputs.get("dt", 0.05)
        P_d0_bar = inputs.get("P_d0_bar", None)

        rho_std = self._model.rho_std
        mdot_load = load_m3h * rho_std / 3600.0      # std m^3/h -> kg/s

        P_d0 = None if P_d0_bar is None else P_d0_bar * 1e5
        return self._model.simulate(P_up, mdot_load, T_up,
                                    P_d0=P_d0, duration_s=dur, dt=dt)

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "P_up_bar": {"unit": "bar", "range": [5, 200]},
                "load_flow_m3_h": {"unit": "std m^3/h", "range": [0, 5000]},
                "T_up_K": {"unit": "K", "range": [240, 330]},
                "duration_s": {"unit": "s"},
                "dt": {"unit": "s"},
                "P_d0_bar": {"unit": "bar"},
            },
            "outputs": {
                "t": "s",
                "P_down_bar": "bar (regulated downstream pressure)",
                "P_set_bar": "bar (setpoint)",
                "valve_travel_frac": "-",
                "flow_std_m3_per_h": "m^3/h",
                "mdot_in_kg_s": "kg/s",
                "mdot_load_kg_s": "kg/s",
                "T_downstream_K": "K (after JT cooling)",
                "JT_cooling_K": "K",
                "is_choked": "bool",
            },
            "source": self._raw.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    info = m.get_info()
    print(f"{info['component_id']} {info['component_name']} -- {info['fidelity']}")
    r = m.predict({"P_up_bar": 50.0, "load_flow_m3_h": 10000.0, "duration_s": 60.0})
    print(f"Setpoint        : {r['P_set_bar'][-1]:.3f} bar")
    print(f"Regulated P_down: {r['P_down_bar'][-1]:.3f} bar")
    print(f"Valve travel    : {r['valve_travel_frac'][-1]*100:.1f} % open")
    print(f"Flow            : {r['flow_std_m3_per_h'][-1]:.1f} std m^3/h")
    print(f"JT cooling      : {r['JT_cooling_K'][-1]:.2f} K  "
          f"(T_down={r['T_downstream_K'][-1]:.2f} K)")
    print(f"Choked          : {bool(r['is_choked'][-1])}")
