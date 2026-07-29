"""
EC051 -- Dye-Sensitized Solar Cell (DSSC) -- F2a Physics-Lumped
Standardised predict() / get_info() interface (mirrors EC001 F2a).
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import DSSC_F2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for the DSSC F2a physics-lumped single-diode model."""

    component_id = "EC051"
    component_name = "Dye-Sensitized Solar Cell (DSSC)"
    fidelity = "F2a -- Physics-Lumped Single-Diode (electrochemical PV) with Thermal ODE"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            self._raw["unit"].update(params)
        self._model = DSSC_F2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        Predict DSSC operating point and dynamic response.

        inputs:
            irradiance_W_m2 : float or callable G(t)  (default 1000.0)
            T_cell_K        : float  initial cell temperature (default 298.15)
            T_amb_K         : float  ambient temperature (default 298.15)
            dt              : float  output step, s (default 2.0)
            duration_s      : float  simulation horizon, s (default 600.0)
            steady_state    : bool   if True, return I-V at (G, T_cell_K) only

        returns: dict with time series (t, temperature, Voc, Isc, Pmp, efficiency)
                 plus the final-state MPP summary and I-V curve.
        """
        G = inputs.get("irradiance_W_m2", 1000.0)
        T0 = inputs.get("T_cell_K", 298.15)
        T_amb = inputs.get("T_amb_K", 298.15)
        dt = inputs.get("dt", 2.0)
        dur = inputs.get("duration_s", 600.0)

        if inputs.get("steady_state", False):
            iv = self._model.iv_curve(float(G) if not callable(G) else G(0.0), T0)
            return {
                "Voc_V": iv["Voc_V"],
                "Isc_A": iv["Isc_A"],
                "Vmp_V": iv["Vmp_V"],
                "Imp_A": iv["Imp_A"],
                "Pmp_W": iv["Pmp_W"],
                "FF": iv["FF"],
                "efficiency": iv["eta"],
                "iv_curve": iv,
            }

        res = self._model.simulate(G, T0=T0, T_amb=T_amb, dt=dt, duration_s=dur)
        iv = res["iv_curve"]
        res["Voc_V"] = iv["Voc_V"]
        res["Isc_A"] = iv["Isc_A"]
        res["Vmp_V"] = iv["Vmp_V"]
        res["Imp_A"] = iv["Imp_A"]
        res["Pmp_W"] = iv["Pmp_W"]
        res["FF"] = iv["FF"]
        return res

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "irradiance_W_m2": {"unit": "W/m2", "range": [0, 1200]},
                "T_cell_K": {"unit": "K", "range": [263.15, 343.15]},
                "T_amb_K": {"unit": "K", "range": [263.15, 333.15]},
                "dt": {"unit": "s", "range": [0.1, 10.0]},
                "duration_s": {"unit": "s", "range": [1.0, 7200.0]},
                "steady_state": {"unit": "bool"},
            },
            "outputs": {
                "t": "s",
                "temperature": "K",
                "Voc": "V",
                "Isc": "A",
                "Pmp": "W",
                "efficiency": "-",
                "iv_curve": "dict (V, I, P arrays + MPP scalars)",
            },
            "source": self._raw.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    # Steady-state operating point at 1 sun, 25 C.
    ss = m.predict({"steady_state": True, "irradiance_W_m2": 1000.0, "T_cell_K": 298.15})
    print(f"\n1-sun MPP: Voc={ss['Voc_V']:.3f} V, Isc={ss['Isc_A']*1e3:.2f} mA, "
          f"Pmp={ss['Pmp_W']*1e3:.2f} mW, FF={ss['FF']:.3f}, eta={ss['efficiency']*100:.2f}%")
    # Low-light / diffuse performance (200 W/m2) -- DSSC strength.
    ll = m.predict({"steady_state": True, "irradiance_W_m2": 200.0, "T_cell_K": 298.15})
    print(f"Low-light (200 W/m2): Voc={ll['Voc_V']:.3f} V, eta={ll['efficiency']*100:.2f}%")
    # Dynamic warm-up.
    r = m.predict({"irradiance_W_m2": 1000.0, "T_cell_K": 298.15, "duration_s": 600.0, "dt": 5.0})
    print(f"Dynamic: T {r['temperature'][0]:.2f} -> {r['temperature'][-1]:.2f} K, "
          f"final Pmp={r['Pmp_W']*1e3:.2f} mW")
