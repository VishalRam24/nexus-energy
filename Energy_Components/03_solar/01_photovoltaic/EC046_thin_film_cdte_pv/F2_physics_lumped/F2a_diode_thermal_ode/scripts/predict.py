"""
EC046 -- Thin-Film CdTe PV -- F2a Physics-Lumped (Single-Diode + Thermal ODE)
Standardised predict() / get_info() interface (mirrors EC001 template).
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import CdTePV_F2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for the CdTe PV F2a physics-lumped model."""

    component_id = "EC046"
    component_name = "Thin-Film CdTe PV"
    fidelity = "F2a -- Single-Diode (Lambert-W) + Lumped Thermal ODE"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            self._raw.update(params)
        self._model = CdTePV_F2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        Two modes:

        Steady-state MPP (default, mode="mpp"):
            irradiance      : float [W/m2]
            cell_temp_c     : float [C]  (if given, used directly)
            T_ambient_c     : float [C]  (used with Faiman if cell_temp_c absent)
            wind_speed      : float [m/s]
        -> {v_mp, i_mp, p_mp, v_oc, i_sc, fill_factor, efficiency, cell_temp_c, iv_curve}

        Dynamic thermal (mode="dynamic"):
            irradiance      : float [W/m2]
            T_ambient_c     : float [C]
            T_cell0_c       : float [C]  (optional initial cell temp)
            wind_speed      : float [m/s]
            duration_s, dt  : floats
        -> time-series {t, temperature, p_mp, v_mp, i_mp, v_oc, i_sc, efficiency}
        """
        mode = inputs.get("mode", "mpp")
        G = inputs.get("irradiance", 1000.0)
        wind = inputs.get("wind_speed", None)

        if mode == "dynamic":
            Ta = inputs.get("T_ambient_c", 25.0)
            T0 = inputs.get("T_cell0_c", None)
            dur = inputs.get("duration_s", 600.0)
            dt = inputs.get("dt", 10.0)
            return self._model.simulate(G, Ta, T_cell0_c=T0, wind=wind,
                                        duration_s=dur, dt=dt)

        # steady-state MPP
        if "cell_temp_c" in inputs:
            Tc = inputs["cell_temp_c"]
        else:
            Ta = inputs.get("T_ambient_c", 25.0)
            Tc = float(self._model.cell_temp_faiman(G, Ta, wind))

        r = self._model.mpp(G, Tc)
        r["cell_temp_c"] = Tc
        r["efficiency"] = self._model.efficiency(G, Tc)
        iv = self._model.iv_curve(G, Tc, n_points=120)
        r["iv_curve"] = {"V": iv["V"], "I": iv["I"], "P": iv["P"]}
        return r

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "mode": {"options": ["mpp", "dynamic"]},
                "irradiance": {"unit": "W/m2", "range": [0, 1200]},
                "cell_temp_c": {"unit": "degC", "range": [-20, 90]},
                "T_ambient_c": {"unit": "degC", "range": [-20, 50]},
                "wind_speed": {"unit": "m/s", "range": [0, 25]},
                "duration_s": {"unit": "s"},
                "dt": {"unit": "s"},
            },
            "outputs": {
                "v_mp": "V", "i_mp": "A", "p_mp": "W",
                "v_oc": "V", "i_sc": "A", "fill_factor": "-",
                "efficiency": "-", "cell_temp_c": "degC",
                "temperature": "degC (dynamic mode time series)",
            },
            "source": self._raw.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    r = m.predict({"irradiance": 1000.0, "cell_temp_c": 25.0})
    print(f"STC: Pmp={r['p_mp']:.1f} W  Vmp={r['v_mp']:.1f} V  "
          f"Imp={r['i_mp']:.3f} A  Voc={r['v_oc']:.1f} V  "
          f"Isc={r['i_sc']:.3f} A  eff={r['efficiency']*100:.2f}%")
    d = m.predict({"mode": "dynamic", "irradiance": 900.0, "T_ambient_c": 30.0,
                   "T_cell0_c": 30.0, "duration_s": 600.0, "dt": 30.0})
    print(f"Dynamic: T_cell {d['temperature'][0]:.1f} -> "
          f"{d['temperature'][-1]:.1f} C, Pmp_final={d['p_mp'][-1]:.1f} W")
