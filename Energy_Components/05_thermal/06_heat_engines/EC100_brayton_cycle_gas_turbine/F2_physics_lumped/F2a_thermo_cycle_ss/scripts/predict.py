"""
EC100 -- Brayton Cycle Gas Turbine -- F2a Physics-Lumped
Standardised predict() / get_info() interface (mirrors EC001 template).
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import Brayton_F2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for the EC100 F2a air-standard Brayton model."""

    component_id = "EC100"
    component_name = "Brayton Cycle Gas Turbine (Simple Cycle)"
    fidelity = "F2a -- Physics-Lumped Air-Standard Brayton Cycle + Spool Dynamics ODE"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            self._raw["unit"].update(params)
        self._model = Brayton_F2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        Evaluate the Brayton cycle and (optionally) the spool transient.

        inputs:
            pressure_ratio       : float (default param PR)
            TIT_K                : float turbine inlet temp limit
            T_inlet_K            : float compressor inlet temp
            regen_effectiveness  : float in [0,1) recuperator
            mdot_air_kg_s        : float air flow
            transient            : bool -- if True, run spool ODE
            load_power_W         : float electrical load for transient
            t_end_s              : float horizon for transient (default 20)
            omega0_rad_s         : float initial shaft speed
        """
        PR = inputs.get("pressure_ratio")
        TIT = inputs.get("TIT_K")
        T1 = inputs.get("T_inlet_K")
        eps = inputs.get("regen_effectiveness")
        mdot = inputs.get("mdot_air_kg_s")

        result = self._model.cycle(PR=PR, TIT=TIT, T1=T1, regen_eps=eps, mdot_air=mdot)
        result["optimal_pressure_ratio"] = self._model.optimal_pressure_ratio(TIT=TIT, T1=T1)

        if inputs.get("transient", False):
            load = inputs.get("load_power_W", result["W_net_W"])
            t_end = inputs.get("t_end_s", 20.0)
            omega0 = inputs.get("omega0_rad_s")
            result["transient"] = self._model.simulate_spool(
                load, omega0=omega0, t_end=t_end, PR=PR, TIT=TIT, T1=T1)
        return result

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "pressure_ratio": {"unit": "-", "range": [2.0, 40.0]},
                "TIT_K": {"unit": "K", "range": [900, 1900]},
                "T_inlet_K": {"unit": "K", "range": [240, 330]},
                "regen_effectiveness": {"unit": "-", "range": [0.0, 0.95]},
                "mdot_air_kg_s": {"unit": "kg/s"},
                "transient": {"unit": "bool"},
                "load_power_W": {"unit": "W"},
                "t_end_s": {"unit": "s"},
            },
            "outputs": {
                "T2_K/T3_K/T4_K": "station temperatures, K",
                "w_net_J_kg": "net specific work, J/kg",
                "eta_thermal": "thermal efficiency, -",
                "eta_carnot": "Carnot ceiling for same T-limits, -",
                "W_net_W": "net shaft power, W",
                "mdot_fuel_kg_s": "fuel mass flow, kg/s",
                "back_work_ratio": "w_compressor / w_turbine, -",
                "transient": "dict: t, omega, rpm, speed_fraction, ...",
            },
            "source": self._raw.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    r = m.predict({"transient": True, "t_end_s": 10.0})
    print(f"\nT2={r['T2_K']:.1f} K  T3={r['T3_K']:.1f} K  T4={r['T4_K']:.1f} K")
    print(f"eta_thermal={r['eta_thermal']:.4f}  (Carnot={r['eta_carnot']:.4f})")
    print(f"W_net={r['W_net_W']/1e6:.2f} MW  mdot_fuel={r['mdot_fuel_kg_s']:.3f} kg/s")
    print(f"back-work ratio={r['back_work_ratio']:.3f}  PR_opt={r['optimal_pressure_ratio']:.1f}")
    tr = r["transient"]
    print(f"spool: rpm {tr['rpm'][0]:.0f} -> {tr['rpm'][-1]:.0f} (success={tr['success']})")
