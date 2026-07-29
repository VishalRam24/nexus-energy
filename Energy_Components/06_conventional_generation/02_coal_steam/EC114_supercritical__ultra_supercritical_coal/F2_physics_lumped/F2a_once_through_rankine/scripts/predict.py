"""
EC114 -- Supercritical / Ultra-Supercritical Coal Plant -- F2a Physics-Lumped
Standardised predict() / get_info() interface (mirrors EC001 template).
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import SupercriticalCoalF2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for EC114 supercritical coal plant F2a model."""

    component_id = "EC114"
    component_name = "Supercritical / Ultra-Supercritical Coal Plant"
    fidelity = "F2a -- Once-Through Supercritical Rankine Cycle with Evaporator Thermal ODE"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            self._raw["unit"].update(params)
        self._model = SupercriticalCoalF2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        Two modes:

        Steady-state (default, mode='cycle'):
            inputs: {"plr": float}  -> full cycle state points + efficiencies.

        Dynamic (mode='transient'):
            inputs: {"mode": "transient", "plr": float-or-list,
                     "dt": float, "duration_s": float, "T_evap0": float}
            -> evaporator thermal transient time series (solve_ivp).
        """
        mode = inputs.get("mode", "cycle")

        if mode == "transient":
            plr = inputs.get("plr", 1.0)
            dt = inputs.get("dt", 10.0)
            dur = inputs.get("duration_s", 1800.0)
            T0 = inputs.get("T_evap0", None)
            if isinstance(plr, (list, tuple)):
                # build a step profile across the horizon
                seq = list(plr)
                seg = dur / max(len(seq), 1)

                def profile(t):
                    idx = min(int(t / seg), len(seq) - 1)
                    return seq[idx]

                return self._model.simulate(profile, dt, dur, T0)
            return self._model.simulate(plr, dt, dur, T0)

        # steady-state cycle
        plr = inputs.get("plr", 1.0)
        return self._model.compute_cycle(plr)

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "plr": {"unit": "-", "range": [0.30, 1.0],
                        "note": "part-load ratio"},
                "mode": {"options": ["cycle", "transient"]},
                "dt": {"unit": "s"},
                "duration_s": {"unit": "s"},
                "T_evap0": {"unit": "degC"},
            },
            "outputs": {
                "eta_net": "-", "eta_cycle": "-", "eta_carnot": "-",
                "P_net_MW": "MW_e", "m_dot": "kg/s (main steam)",
                "coal_rate_kgs": "kg/s", "co2_rate_kgs": "kg/s",
                "co2_intensity_g_per_kwh": "g/kWh",
                "state_points": "dict h[kJ/kg],T[degC],P[bar]",
                "(transient)": "t, T_evap, eta_net, P_net_MW, Q_furnace_MW, Q_steam_MW",
            },
            "source": self._raw.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    r = m.predict({"plr": 1.0})
    print(f"\nFull-load: eta_net={r['eta_net']:.4f}  eta_cycle={r['eta_cycle']:.4f} "
          f"(Carnot={r['eta_carnot']:.4f})")
    print(f"  P_net={r['P_net_MW']:.0f} MW  coal={r['coal_rate_kgs']:.1f} kg/s  "
          f"CO2={r['co2_intensity_g_per_kwh']:.0f} g/kWh")
    rt = m.predict({"mode": "transient", "plr": [1.0, 0.6, 1.0],
                    "dt": 30.0, "duration_s": 900.0})
    print(f"\nTransient: T_evap {rt['T_evap'][0]:.1f} -> {rt['T_evap'][-1]:.1f} C, "
          f"{len(rt['t'])} steps")
