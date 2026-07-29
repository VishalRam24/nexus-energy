"""
EC112 -- Micro Gas Turbine -- F2a Recuperated Brayton Cycle (Physics-Lumped)
Standardised predict() / get_info() interface (mirrors EC001 F2a).
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import MicroGasTurbineF2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for the EC112 recuperated Brayton micro gas turbine."""

    component_id = "EC112"
    component_name = "Micro Gas Turbine"
    fidelity = "F2a -- Recuperated Brayton Cycle (Physics-Lumped) with Shaft + Recuperator ODE"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            self._raw["unit"].update(params)
        self._model = MicroGasTurbineF2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        Two modes:
          mode="steady" (default): solve the recuperated Brayton cycle and return
              station temperatures, works, efficiencies, power.
          mode="transient": run the lumped shaft + recuperator ODE.

        Steady inputs:
            PLR : float (default 1.0)          part-load ratio
            T_amb_K : float (default ISO 288.15)
        Transient inputs:
            fuel_fraction : float (default 1.0)
            P_load_kw : float or None
            T_amb_K : float
            omega0_frac : float (default 1.0)
            T_rec0_K : float or None (default cold start at ambient)
            dt : float (default 0.5)
            duration_s : float (default 120.0)
        """
        mode = inputs.get("mode", "steady")
        T_amb = inputs.get("T_amb_K", None)

        if mode == "transient":
            return self._model.simulate(
                fuel_fraction=inputs.get("fuel_fraction", 1.0),
                P_load_kw=inputs.get("P_load_kw", None),
                T_amb=T_amb,
                omega0_frac=inputs.get("omega0_frac", 1.0),
                T_rec0=inputs.get("T_rec0_K", None),
                dt=inputs.get("dt", 0.5),
                duration_s=inputs.get("duration_s", 120.0),
            )

        PLR = inputs.get("PLR", 1.0)
        if PLR >= 0.999:
            c = self._model.cycle(T_amb=T_amb)
            c["PLR"] = 1.0
            c["speed_fraction"] = 1.0
            return c
        return self._model.partload_state(PLR, T_amb=T_amb)

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "mode": {"unit": "-", "options": ["steady", "transient"]},
                "PLR": {"unit": "-", "range": [0.3, 1.0]},
                "T_amb_K": {"unit": "K", "range": [248.15, 323.15]},
                "fuel_fraction": {"unit": "-", "range": [0.3, 1.1]},
                "P_load_kw": {"unit": "kW_e"},
                "dt": {"unit": "s"},
                "duration_s": {"unit": "s"},
            },
            "outputs": {
                "steady": "T1..T5, w_comp/w_turb/w_net, q_in, eta_thermal, "
                          "eta_electrical, eta_carnot, P_el_W, Q_fuel_W, T_exhaust_K",
                "transient": "t, omega, speed_rpm, T_recup, P_el_kw, "
                             "eta_electrical, eta_carnot, T_exhaust_K, fuel_kw",
            },
            "source": self._raw.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    s = m.predict({"mode": "steady", "PLR": 1.0})
    print(f"\n[Steady, full load] eta_el={s['eta_electrical']*100:.1f}%  "
          f"(Carnot={s['eta_carnot']*100:.1f}%)  P_el={s['P_el_W']/1e3:.1f} kW  "
          f"T_exhaust={s['T_exhaust_K']:.0f} K")
    nr = m._model.cycle_no_recuperator()
    print(f"[No recuperator]    eta_el={nr['eta_electrical']*100:.1f}%  "
          f"-> recuperator gain = {(s['eta_electrical']-nr['eta_electrical'])*100:.1f} pts")
    p = m.predict({"mode": "steady", "PLR": 0.5})
    print(f"[Steady, 50% load]  eta_el={p['eta_electrical']*100:.1f}%  "
          f"P_el={p['P_el_W']/1e3:.1f} kW  speed={p['speed_fraction']*100:.0f}%")
    tr = m.predict({"mode": "transient", "duration_s": 60.0, "dt": 2.0})
    print(f"[Transient 60 s]    T_recup {tr['T_recup'][0]:.0f}->{tr['T_recup'][-1]:.0f} K  "
          f"P_el_final={tr['P_el_kw'][-1]:.1f} kW")
