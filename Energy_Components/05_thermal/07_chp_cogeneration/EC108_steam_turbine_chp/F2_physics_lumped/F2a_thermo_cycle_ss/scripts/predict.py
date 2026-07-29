"""
EC108 -- Steam Turbine CHP -- F2a Physics-Lumped Thermo-Cycle
Standardised predict() / get_info() interface (mirrors EC001 F2a).
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import SteamTurbineCHPF2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for the EC108 steam-turbine CHP F2a model."""

    component_id = "EC108"
    component_name = "Steam Turbine CHP (back-pressure / extraction)"
    fidelity = "F2a -- Physics-Lumped Rankine Cycle + Boiler Thermal ODE"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            self._raw["unit"].update(params)
        self._model = SteamTurbineCHPF2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        inputs:
            PLR : float                part-load ratio (default 1.0)
            transient : bool           if True, run lumped thermal ODE
            T0_C : float               initial boiler temperature [degC]
            duration_s : float         transient horizon (default 1800 s)
            dt : float                 output step (default 5 s)

        Returns the steady-state cycle dict; if transient=True also embeds a
        'transient' key with the boiler thermal time series.
        """
        PLR = inputs.get("PLR", 1.0)
        result = self._model.steady_state(PLR)

        if inputs.get("transient", False):
            tr = self._model.simulate(
                PLR,
                T0_C=inputs.get("T0_C", None),
                duration_s=inputs.get("duration_s", 1800.0),
                dt=inputs.get("dt", 5.0),
            )
            result["transient"] = tr

        return result

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "PLR": {"unit": "-", "range": [0.3, 1.0]},
                "transient": {"unit": "bool"},
                "T0_C": {"unit": "degC"},
                "duration_s": {"unit": "s"},
                "dt": {"unit": "s"},
            },
            "outputs": {
                "P_el_kw": "kW_e",
                "Q_useful_kw": "kW_th",
                "Q_fuel_kw": "kW",
                "eta_el": "-",
                "eta_th": "-",
                "eta_total": "-",
                "power_to_heat": "-",
                "HPR": "-",
                "eta_carnot": "-",
                "transient": "dict of time series (if requested)",
            },
            "source": self._raw["unit"].get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    r = m.predict({"PLR": 1.0})
    print(
        f"\nP_el = {r['P_el_kw']:.0f} kW_e | Q_useful = {r['Q_useful_kw']:.0f} kW_th"
        f"\neta_el = {r['eta_el']:.3f} | eta_th = {r['eta_th']:.3f} | "
        f"eta_total = {r['eta_total']:.3f}"
        f"\npower-to-heat = {r['power_to_heat']:.3f} | "
        f"eta_carnot = {r['eta_carnot']:.3f}"
    )
    tr = m.predict({"PLR": 1.0, "transient": True, "T0_C": 120.0,
                    "duration_s": 1200.0, "dt": 10.0})["transient"]
    print(f"\nTransient: T_boiler {tr['T_boiler_C'][0]:.1f} -> "
          f"{tr['T_boiler_C'][-1]:.1f} degC over {tr['t'][-1]:.0f} s "
          f"(success={tr['success']})")
