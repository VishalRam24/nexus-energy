"""
EC053 -- Thermophotovoltaic (TPV) -- F2a Spectral Radiative
Standardised predict() / get_info() interface.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import TPV_F2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for TPV F2a spectral radiative physics-lumped model."""

    component_id = "EC053"
    component_name = "Thermophotovoltaic (TPV)"
    fidelity = "F2a -- Spectral Radiative + Single-Diode I-V + Cell Thermal ODE"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            self._raw["unit"].update(params)
        self._model = TPV_F2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        Run the dynamic TPV simulation (cell thermal ODE + electrical output).

        inputs:
            T_emitter_K : float (or callable f(t)->K for time-varying emitter)
            T_cell0_K   : float  initial cell temperature (default 300.0)
            dt          : float  output step [s] (default 0.1)
            duration_s  : float  simulation horizon [s] (default 60.0)

        returns dict of time-series arrays plus a steady MPP summary at the
        initial emitter temperature.
        """
        Te = inputs.get("T_emitter_K", 1500.0)
        Tc0 = inputs.get("T_cell0_K", 300.0)
        dt = inputs.get("dt", 0.1)
        dur = inputs.get("duration_s", 60.0)

        res = self._model.simulate(Te, Tc0, dt, dur)
        # add a static MPP summary at the (initial) emitter T for convenience
        Te0 = Te(0.0) if callable(Te) else Te
        res["mpp_summary"] = self._model.efficiencies(Te0, Tc0)
        return res

    def get_info(self) -> dict:
        u = self._raw["unit"]
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "T_emitter_K": {"unit": "K", "range": [800, 2400]},
                "T_cell0_K": {"unit": "K", "range": [280, 400]},
                "dt": {"unit": "s", "range": [0.001, 1.0]},
                "duration_s": {"unit": "s", "range": [0.1, 600.0]},
            },
            "outputs": {
                "t": "s",
                "T_cell": "K",
                "T_emitter": "K",
                "P_elec_W": "W",
                "Vmp": "V",
                "eta_system": "-",
                "eta_spectral": "-",
                "mpp_summary": "dict (Voc, Jsc, FF, Pmp, efficiencies)",
            },
            "device": {
                "bandgap_eV": u["E_g_eV"]["value"],
                "cell_area_m2": u["A_cell"]["value"],
                "emitter": "selective/blackbody (SiC-like)",
            },
            "source": self._raw.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    info = m.get_info()
    print(f"{info['component_id']} {info['component_name']} -- {info['fidelity']}")
    r = m.predict({"T_emitter_K": 1500.0, "T_cell0_K": 300.0, "dt": 1.0, "duration_s": 30.0})
    s = r["mpp_summary"]
    print(f"Steady MPP @1500K: P={s['P_elec_W']*1e3:.2f} mW, Voc={s['Voc']:.3f} V, "
          f"FF={s['FF']:.3f}, eta_sys={s['eta_system']*100:.1f}%, "
          f"eta_spec={s['eta_spectral']*100:.1f}%")
    print(f"Final cell T: {r['T_cell'][-1]:.2f} K, final P: {r['P_elec_W'][-1]*1e3:.2f} mW")
