"""
EC052 -- Bifacial PV Module -- F2a Physics-Lumped (Single-Diode + Thermal ODE)
Standardised predict() / get_info() interface (mirrors EC001 template).
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import BifacialPV_F2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for the EC052 bifacial PV F2a physics-lumped model."""

    component_id = "EC052"
    component_name = "Bifacial PV Module"
    fidelity = "F2a -- Bifacial Single-Diode (Lambert-W) + Lumped Thermal ODE"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            self._raw.update(params)
        self._model = BifacialPV_F2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        Run transient bifacial PV simulation.

        inputs:
            G_front_W_m2 : float (or callable t->float)  front-side irradiance
            T_amb_C      : float (or callable)           ambient temperature [degC]
            wind_speed_m_s : float (or callable)         wind speed
            G_rear_W_m2  : float or None                 explicit rear irradiance
            albedo       : float or None                 ground albedo (rear = albedo*G_front*F_view)
            T_cell0_C    : float or None                 initial cell temp (default = T_amb)
            dt           : float (default 60.0)          output step [s]
            duration_s   : float (default 3600.0)        sim length [s]
        """
        G_front = inputs.get("G_front_W_m2", 1000.0)
        T_amb = inputs.get("T_amb_C", 25.0)
        v_wind = inputs.get("wind_speed_m_s", 1.0)
        G_rear = inputs.get("G_rear_W_m2", None)
        albedo = inputs.get("albedo", None)
        T0 = inputs.get("T_cell0_C", None)
        dt = inputs.get("dt", 60.0)
        dur = inputs.get("duration_s", 3600.0)

        return self._model.simulate(
            G_front, T_amb_C=T_amb, v_wind=v_wind,
            G_rear=G_rear, albedo=albedo, T_cell0_C=T0,
            dt=dt, duration_s=dur)

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "G_front_W_m2": {"unit": "W/m2", "range": [0, 1200]},
                "T_amb_C": {"unit": "degC", "range": [-20, 50]},
                "wind_speed_m_s": {"unit": "m/s", "range": [0, 25]},
                "G_rear_W_m2": {"unit": "W/m2", "range": [0, 600]},
                "albedo": {"unit": "-", "range": [0, 0.9]},
                "T_cell0_C": {"unit": "degC"},
                "dt": {"unit": "s"},
                "duration_s": {"unit": "s"},
            },
            "outputs": {
                "t": "s",
                "temperature_C": "degC",
                "v_mp": "V", "i_mp": "A", "p_mp": "W",
                "v_oc": "V", "i_sc": "A",
                "efficiency": "-",
                "G_effective": "W/m2",
                "bifacial_gain": "-",
            },
            "source": self._raw.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    r = m.predict({"G_front_W_m2": 900.0, "albedo": 0.5, "T_amb_C": 25.0,
                   "wind_speed_m_s": 1.0, "duration_s": 1800.0, "dt": 60.0})
    print(f"Final T_cell: {r['temperature_C'][-1]:.2f} C, "
          f"P_mp: {r['p_mp'][-1]:.1f} W, "
          f"eff: {r['efficiency'][-1]*100:.2f} %, "
          f"bifacial_gain: {r['bifacial_gain'][-1]*100:.2f} %")
