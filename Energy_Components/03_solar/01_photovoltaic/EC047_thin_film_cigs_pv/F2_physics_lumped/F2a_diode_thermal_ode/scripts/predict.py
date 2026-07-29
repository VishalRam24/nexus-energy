"""
EC047 -- Thin-Film CIGS PV -- F2a Physics-Lumped (single-diode + thermal ODE)
Standardised predict() / get_info() interface (mirrors EC001 template).
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import CIGSPvF2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for CIGS PV F2a physics-lumped model."""

    component_id = "EC047"
    component_name = "Thin-Film CIGS PV"
    fidelity = "F2a -- Physics-Lumped Single-Diode (De Soto/Lambert-W) + Thermal ODE"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            self._raw["unit"].update(params)
        self._model = CIGSPvF2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        Modes:
          mode="mpp"      (default) -- steady MPP at given G + cell temperature.
            inputs: irradiance [W/m2], cell_temperature_c [degC]
            returns: v_mp, i_mp, p_mp, v_oc, i_sc, fill_factor, efficiency
          mode="iv"       -- full I-V/P-V curve.
            inputs: irradiance, cell_temperature_c, n (points)
            returns: V, I, P arrays
          mode="transient" -- lumped thermal ODE over time.
            inputs: irradiance, T_ambient_c, wind, T_cell0_c, dt, duration_s
            returns: time-series dict
        """
        mode = inputs.get("mode", "mpp")
        G = inputs.get("irradiance", 1000.0)

        if mode == "transient":
            r = self._model.simulate(
                irradiance=G,
                T_amb_c=inputs.get("T_ambient_c", 25.0),
                wind=inputs.get("wind", 1.0),
                T_cell0_c=inputs.get("T_cell0_c", None),
                dt=inputs.get("dt", 60.0),
                duration_s=inputs.get("duration_s", 3600.0),
            )
            return r

        Tc = inputs.get("cell_temperature_c", 25.0)
        T_K = Tc + 273.15

        if mode == "iv":
            V, I, P = self._model.iv_curve(G, T_K, n=inputs.get("n", 200))
            return {"V": V, "I": I, "P": P}

        # default: mpp
        r = self._model.mpp(G, T_K)
        r["efficiency"] = self._model.efficiency(G, T_K)
        return r

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "mode": {"options": ["mpp", "iv", "transient"]},
                "irradiance": {"unit": "W/m2", "range": [0, 1200]},
                "cell_temperature_c": {"unit": "degC", "range": [-20, 95]},
                "T_ambient_c": {"unit": "degC", "range": [-20, 50]},
                "wind": {"unit": "m/s", "range": [0, 20]},
                "dt": {"unit": "s"},
                "duration_s": {"unit": "s"},
            },
            "outputs": {
                "v_mp": "V", "i_mp": "A", "p_mp": "W",
                "v_oc": "V", "i_sc": "A", "fill_factor": "-",
                "efficiency": "-",
                "transient": "dict of time-series arrays (t, T_cell_c, power_W, ...)",
            },
            "source": self._raw.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    r = m.predict({"mode": "mpp", "irradiance": 1000.0, "cell_temperature_c": 25.0})
    print(f"STC MPP: P_mp={r['p_mp']:.1f} W, V_mp={r['v_mp']:.1f} V, "
          f"I_mp={r['i_mp']:.2f} A, Voc={r['v_oc']:.1f} V, "
          f"FF={r['fill_factor']:.3f}, eff={r['efficiency']*100:.1f}%")
    tr = m.predict({"mode": "transient", "irradiance": 900.0,
                    "T_ambient_c": 30.0, "wind": 1.0, "dt": 60.0,
                    "duration_s": 3600.0})
    print(f"Transient: T_cell {tr['T_cell_c'][0]:.1f} -> {tr['T_cell_c'][-1]:.1f} degC, "
          f"P {tr['power_W'][0]:.1f} -> {tr['power_W'][-1]:.1f} W")
