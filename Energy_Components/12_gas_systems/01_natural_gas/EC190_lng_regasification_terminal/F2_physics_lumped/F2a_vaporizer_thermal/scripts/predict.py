"""
EC190 -- LNG Regasification Terminal -- F2a Physics-Lumped Vaporizer Thermal
Standardised predict() / get_info() interface (mirrors EC001 template).
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import LNGRegasF2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for EC190 LNG regas F2a vaporizer thermal model."""

    component_id = "EC190"
    component_name = "LNG Regasification Terminal"
    fidelity = "F2a -- Physics-Lumped Vaporizer Thermal Transient (ODE)"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            # allow shallow override of vaporizer/pump/lng sub-dicts
            for k, v in params.items():
                if k in self._raw and isinstance(self._raw[k], dict):
                    self._raw[k].update(v)
                else:
                    self._raw[k] = v
        self._model = LNGRegasF2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        Run the lumped vaporizer thermal transient.

        inputs:
            sendout_rate_ton_per_h : float (or callable(t)->ton/h)   default 500
            T_metal0_K             : float  initial metal temp (default cold-start)
            T_heat_source_K        : float (or callable)  default param (seawater 15 C)
            T_ambient_K            : float  default 288.15
            dt                     : float  default 10.0
            duration_s             : float  default 3600.0
            T_sendout_K            : float  default param (5 C)
        """
        m = inputs.get("sendout_rate_ton_per_h", 500.0)
        T_metal0 = inputs.get("T_metal0_K", None)
        T_src = inputs.get("T_heat_source_K", None)
        T_amb = inputs.get("T_ambient_K", 288.15)
        dt = inputs.get("dt", 10.0)
        dur = inputs.get("duration_s", 3600.0)
        Ts = inputs.get("T_sendout_K", None)

        return self._model.simulate(
            m, T_metal0=T_metal0, T_heat_source_K=T_src,
            T_ambient_K=T_amb, dt=dt, duration_s=dur, T_sendout=Ts,
        )

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "sendout_rate_ton_per_h": {"unit": "ton/h", "range": [10, 5000]},
                "T_metal0_K": {"unit": "K"},
                "T_heat_source_K": {"unit": "K", "range": [275, 350]},
                "T_ambient_K": {"unit": "K", "range": [250, 320]},
                "dt": {"unit": "s"},
                "duration_s": {"unit": "s"},
                "T_sendout_K": {"unit": "K"},
            },
            "outputs": {
                "t": "s",
                "T_metal": "K",
                "Q_source_W": "W",
                "Q_process_W": "W",
                "Q_demand_W": "W",
                "sendout_kg_s": "kg/s",
                "pump_W": "W",
                "cold_exergy_W": "W",
                "energy_balance": "dict of J (E_source, E_process, E_stored, residual)",
            },
            "source": self._raw.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    r = m.predict({"sendout_rate_ton_per_h": 500.0, "duration_s": 3600.0, "dt": 30.0})
    eb = r["energy_balance"]
    print(f"Final T_metal: {r['T_metal'][-1]:.2f} K")
    print(f"Steady Q_process: {r['Q_process_W'][-1]/1e6:.2f} MW, "
          f"demand: {r['Q_demand_W'][-1]/1e6:.2f} MW")
    print(f"Pump: {r['pump_W'][-1]/1e6:.3f} MW, "
          f"Cold exergy potential: {r['cold_exergy_W'][-1]/1e6:.2f} MW")
    print(f"Energy balance residual: {eb['residual_J']:.3e} J "
          f"(source {eb['E_source_J']:.3e}, process {eb['E_process_J']:.3e}, "
          f"stored {eb['E_stored_J']:.3e})")
