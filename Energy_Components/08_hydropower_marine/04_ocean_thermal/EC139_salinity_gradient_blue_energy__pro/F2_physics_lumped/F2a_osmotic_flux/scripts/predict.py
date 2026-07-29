"""
EC139 -- Salinity Gradient Blue Energy (PRO) -- F2a Osmotic Flux
Standardised predict() / get_info() interface.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import SalinityGradientPRO_F2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")

_BAR = 1.0e5  # Pa per bar


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for PRO F2a osmotic-flux lumped model."""

    component_id = "EC139"
    component_name = "Salinity Gradient Blue Energy (PRO)"
    fidelity = "F2a -- Osmotic Flux Lumped Module (Jw/Js + CP + dilution ODE)"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            self._raw["unit"].update(params)
        self._model = SalinityGradientPRO_F2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        Run a lumped PRO module simulation.

        inputs:
            delta_P_bar     : float  hydraulic pressure on draw side [bar]
                              (default: optimal DeltaP = Delta_pi/2)
            C_draw_g_per_L  : float  draw (seawater) salinity [g/L] (default 35)
            C_feed_g_per_L  : float  feed (river) salinity [g/L]   (default 0.5)
            T_K             : float  temperature [K]               (default 298.15)
            dt              : float  output timestep [s]           (default 1.0)
            duration_s      : float  simulation length [s]         (default 600)
        """
        C_draw = inputs.get("C_draw_g_per_L", self._model.C_draw0)
        C_feed = inputs.get("C_feed_g_per_L", self._model.C_feed0)
        T_K = inputs.get("T_K", self._model.T_K)
        dt = inputs.get("dt", 1.0)
        dur = inputs.get("duration_s", 600.0)

        if "delta_P_bar" in inputs:
            dP = float(inputs["delta_P_bar"]) * _BAR
        else:
            dP, _ = self._model.optimal_delta_P(C_draw, C_feed, T_K)

        res = self._model.simulate(dP, C_draw, C_feed, T_K, dt, dur)
        # convenient scalar summary at module steady state (last point)
        res["delta_P_bar"] = dP / _BAR
        res["power_density_final_Wm2"] = float(res["power_density"][-1])
        res["P_net_final_W"] = float(res["P_net_W"][-1])
        res["Jw_final_LMH"] = float(res["Jw_LMH"][-1])
        return res

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "delta_P_bar": {"unit": "bar", "range": [0, 30]},
                "C_draw_g_per_L": {"unit": "g/L", "range": [20, 70]},
                "C_feed_g_per_L": {"unit": "g/L", "range": [0.1, 5]},
                "T_K": {"unit": "K", "range": [278.15, 313.15]},
                "dt": {"unit": "s"},
                "duration_s": {"unit": "s"},
            },
            "outputs": {
                "t": "s",
                "C_draw_gL": "g/L",
                "Jw": "m/s",
                "Jw_LMH": "L/(m2.h)",
                "Js": "mol/(m2.s)",
                "power_density": "W/m2",
                "P_turbine_W": "W",
                "P_pump_W": "W",
                "P_net_W": "W",
            },
            "source": self._raw.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    # optimal-DeltaP run at seawater/river conditions
    r = m.predict({"duration_s": 300.0, "dt": 5.0})
    print(f"Optimal DeltaP = {r['delta_P_bar']:.2f} bar")
    print(f"Peak power density = {r['power_density_final_Wm2']:.2f} W/m2")
    print(f"Water flux Jw = {r['Jw_final_LMH']:.2f} L/m2/h")
    print(f"Net module power = {r['P_net_final_W']:.1f} W "
          f"(area {m._model.A_mem:.0f} m2)")
