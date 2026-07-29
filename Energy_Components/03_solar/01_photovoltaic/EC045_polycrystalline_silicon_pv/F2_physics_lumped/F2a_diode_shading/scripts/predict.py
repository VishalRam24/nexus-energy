"""
EC045 -- Polycrystalline Silicon PV -- F2a Physics-Lumped Single-Diode
Standardised predict() / get_info() interface (mirrors EC001 template).
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import PolySiPVF2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for EC045 Poly-Si PV F2a physics-lumped model."""

    component_id = "EC045"
    component_name = "Polycrystalline Silicon PV"
    fidelity = "F2a -- Physics-Lumped Single-Diode (De Soto) + Thermal ODE + Partial Shading"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            self._raw["unit"].update(params)
        self._model = PolySiPVF2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        Two modes:
          mode="steady" (default): single-point MPP at given (G, T_cell).
              irradiance_W_m2 : float
              cell_temp_C     : float (if omitted, derived from T_ambient via NOCT)
              T_ambient_C     : float (default 25)
              wind_m_s        : float (default 1.0)
              shade_fraction  : float in [0,1] (default 0) -- shades one substring
          mode="dynamic": integrate the cell-temperature ODE over time.
              irradiance_W_m2 / T_ambient_C / wind_m_s may be scalars
              dt, duration_s, shade_fraction as above
        """
        mode = inputs.get("mode", "steady")
        G = inputs.get("irradiance_W_m2", 1000.0)
        T_amb = inputs.get("T_ambient_C", 25.0)
        wind = inputs.get("wind_m_s", 1.0)
        shade = inputs.get("shade_fraction", 0.0)

        if mode == "dynamic":
            dt = inputs.get("dt", 60.0)
            dur = inputs.get("duration_s", 3600.0)
            T_cell0 = inputs.get("cell_temp_C", None)
            res = self._model.simulate(
                G, T_amb=T_amb, wind=wind, dt=dt, duration_s=dur,
                T_cell0=T_cell0, shade_fraction=shade,
            )
            return res

        # steady-state single point
        T_cell = inputs.get("cell_temp_C", None)
        if T_cell is None:
            T_cell = self._model._steady_cell_temp(G, T_amb, wind)
        if shade > 0.0:
            r = self._model.mpp_partial_shade(G, T_cell, shade)
        else:
            r = self._model.mpp(G, T_cell)
        eff = (r["p_mp"] / (G * self._model.area)) if G > 1.0 else 0.0
        return {
            "v_mp": r["v_mp"],
            "i_mp": r["i_mp"],
            "p_mp": r["p_mp"],
            "v_oc": r["v_oc"],
            "i_sc": r["i_sc"],
            "efficiency": eff,
            "cell_temp_C": float(T_cell),
        }

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "mode": {"options": ["steady", "dynamic"]},
                "irradiance_W_m2": {"unit": "W/m2", "range": [0, 1200]},
                "cell_temp_C": {"unit": "degC", "range": [-20, 90]},
                "T_ambient_C": {"unit": "degC", "range": [-20, 50]},
                "wind_m_s": {"unit": "m/s", "range": [0, 25]},
                "shade_fraction": {"unit": "-", "range": [0, 1]},
                "dt": {"unit": "s"},
                "duration_s": {"unit": "s"},
            },
            "outputs": {
                "v_mp": "V", "i_mp": "A", "p_mp": "W",
                "v_oc": "V", "i_sc": "A", "efficiency": "-",
                "cell_temp_C": "degC",
                "(dynamic adds)": "t[s], T_cell[degC], time-series arrays",
            },
            "source": self._raw.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    print()
    r = m.predict({"irradiance_W_m2": 1000.0, "cell_temp_C": 25.0})
    print(f"STC: P_mp={r['p_mp']:.1f} W  V_mp={r['v_mp']:.2f} V  "
          f"I_mp={r['i_mp']:.2f} A  V_oc={r['v_oc']:.2f} V  "
          f"I_sc={r['i_sc']:.2f} A  eff={r['efficiency']*100:.2f}%")
    rs = m.predict({"irradiance_W_m2": 1000.0, "cell_temp_C": 25.0, "shade_fraction": 1.0})
    print(f"1 substring fully shaded: P_mp={rs['p_mp']:.1f} W "
          f"(vs {r['p_mp']:.1f} W unshaded)")
    rd = m.predict({"mode": "dynamic", "irradiance_W_m2": 900.0,
                    "T_ambient_C": 30.0, "wind_m_s": 1.0,
                    "dt": 60.0, "duration_s": 1800.0})
    print(f"Dynamic 30 min: T_cell {rd['T_cell'][0]:.1f} -> "
          f"{rd['T_cell'][-1]:.1f} C, P_mp {rd['p_mp'][-1]:.1f} W")
