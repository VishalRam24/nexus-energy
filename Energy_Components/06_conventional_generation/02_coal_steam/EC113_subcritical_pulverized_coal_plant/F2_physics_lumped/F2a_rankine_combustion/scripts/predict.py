"""
EC113 -- Subcritical Pulverized Coal Plant -- F2a Physics-Lumped
Standardised predict() / get_info() interface (mirrors EC001 template).
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import SubcriticalCoalF2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for the EC113 F2a combustion + Rankine + drum model."""

    component_id = "EC113"
    component_name = "Subcritical Pulverized Coal Plant"
    fidelity = "F2a -- Physics-Lumped Combustion + Subcritical Rankine Cycle + Drum Thermal ODE"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            self._raw["unit"].update(params)
        self._model = SubcriticalCoalF2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        Run the lumped drum thermal transient + steady-state plant metrics.

        inputs:
            part_load_ratio : float or callable plr(t)  (default 1.0)
            T0_drum_K       : float  (initial drum temp; default cold air)
            dt              : float  (s; default 20.0)
            duration_s      : float  (s; default 8000.0)
        returns dict with time-series + scalar design metrics:
            t, T_drum, T_sat_drum, plr, power_net_mw, coal_rate_kgs,
            co2_rate_kgs, steam_rate_kgs (arrays);
            eta_net, eta_boiler, eta_cycle, eta_carnot (scalars).
        """
        plr = inputs.get("part_load_ratio", 1.0)
        T0 = inputs.get("T0_drum_K", None)
        dt = inputs.get("dt", 20.0)
        dur = inputs.get("duration_s", 8000.0)
        return self._model.simulate(plr=plr, T0_K=T0, dt=dt, duration_s=dur)

    def steady_state(self, plr: float = 1.0) -> dict:
        """Convenience: steady-state design-point metrics (no time integration)."""
        m = self._model
        eta_net, eta_b, eta_c = m.net_efficiency()
        return {
            "part_load_ratio": plr,
            "power_net_mw": m.P_rated * plr,
            "eta_net": eta_net,
            "eta_boiler": eta_b,
            "eta_cycle": eta_c,
            "eta_carnot": m.carnot_efficiency(),
            "coal_rate_kgs": m.coal_rate_kgs(plr),
            "co2_rate_kgs": m.co2_rate_kgs(plr),
            "co2_intensity_g_per_kwh": m.co2_intensity_g_per_kwh(plr),
            "steam_rate_kgs": m.steam_flow_kgs(plr),
        }

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "part_load_ratio": {"unit": "-", "range": [0.30, 1.0]},
                "T0_drum_K": {"unit": "K", "range": [290, 640]},
                "dt": {"unit": "s", "range": [0.1, 60]},
                "duration_s": {"unit": "s", "range": [1, 100000]},
            },
            "outputs": {
                "t": "s",
                "T_drum": "K",
                "T_sat_drum": "K",
                "plr": "-",
                "power_net_mw": "MW_e",
                "coal_rate_kgs": "kg/s",
                "co2_rate_kgs": "kg/s",
                "steam_rate_kgs": "kg/s",
                "eta_net": "-",
                "eta_boiler": "-",
                "eta_cycle": "-",
                "eta_carnot": "-",
            },
            "source": self._raw.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    ss = m.steady_state(1.0)
    print(f"\nDesign point: P={ss['power_net_mw']:.0f} MW_e, "
          f"eta_net={ss['eta_net']*100:.1f}% "
          f"(boiler {ss['eta_boiler']*100:.1f}%, cycle {ss['eta_cycle']*100:.1f}%), "
          f"Carnot {ss['eta_carnot']*100:.1f}%")
    print(f"Coal {ss['coal_rate_kgs']:.1f} kg/s, steam {ss['steam_rate_kgs']:.0f} kg/s, "
          f"CO2 {ss['co2_intensity_g_per_kwh']:.0f} g/kWh")
    r = m.predict({"part_load_ratio": 1.0, "T0_drum_K": 320.0,
                   "dt": 20.0, "duration_s": 8000.0})
    print(f"\nDrum warm-up: T0={r['T_drum'][0]:.1f} K -> "
          f"T_final={r['T_drum'][-1]:.1f} K (T_sat={r['T_sat_drum'][0]:.1f} K), "
          f"steam_final={r['steam_rate_kgs'][-1]:.0f} kg/s")
