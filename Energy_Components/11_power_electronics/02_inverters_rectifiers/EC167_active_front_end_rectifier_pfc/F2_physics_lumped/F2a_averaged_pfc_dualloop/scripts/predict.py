"""
EC167 -- Active Front End Rectifier / PFC -- F2a Averaged Boost-PFC, Dual-Loop
Standardised predict() / get_info() interface (mirrors EC001 template).
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import BoostPFC_F2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for the boost-PFC averaged dual-loop model."""

    component_id = "EC167"
    component_name = "Active Front End Rectifier / PFC (boost)"
    fidelity = "F2a -- Averaged Boost-PFC with Dual-Loop Control (DC-link cap ODE)"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            self._raw["unit"].update(params)
        self._model = BoostPFC_F2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        Run the averaged boost-PFC simulation and return steady-state KPIs + waveforms.

        inputs:
            V_line_rms : float  RMS mains voltage [V]            (default 230)
            v_ref      : float  DC-link voltage setpoint [V]     (default 400)
            P_load     : float  DC load power [W]                (default 3000)
            duration_s : float  sim horizon [s]                  (default 0.12)
            n_points   : int    output samples                   (default 4000)

        returns dict:
            power_factor, thd_current, efficiency, v_dc_mean, V_peak,
            p_loss_w, loss_breakdown, and full waveform arrays under 'waveforms'.
        """
        Vrms = inputs.get("V_line_rms", None)
        vref = inputs.get("v_ref", None)
        Pld = inputs.get("P_load", None)
        dur = inputs.get("duration_s", 0.12)
        npts = int(inputs.get("n_points", 4000))

        res = self._model.simulate(Vrms, vref, Pld, duration_s=dur, n_points=npts)
        kpis = self._model.summary(res)
        kpis["waveforms"] = {
            "t": res["t"],
            "i_line": res["i_line"],
            "v_line": res["v_line"],
            "v_dc": res["v_dc"],
            "i_L": res["i_L"],
        }
        return kpis

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "V_line_rms": {"unit": "V", "range": [90, 265]},
                "v_ref": {"unit": "V", "range": [360, 450]},
                "P_load": {"unit": "W", "range": [0, 3300]},
                "duration_s": {"unit": "s", "range": [0.02, 2.0]},
                "n_points": {"unit": "-"},
            },
            "outputs": {
                "power_factor": "-",
                "thd_current": "fraction",
                "efficiency": "-",
                "v_dc_mean": "V",
                "V_peak": "V",
                "p_loss_w": "W",
                "loss_breakdown": "dict of W",
                "waveforms": "dict of arrays",
            },
            "source": self._raw.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    r = m.predict({"P_load": 3000.0, "duration_s": 0.12})
    print(f"PF={r['power_factor']:.4f}  THD={r['thd_current']*100:.2f}%  "
          f"eta={r['efficiency']*100:.2f}%  V_dc={r['v_dc_mean']:.1f} V "
          f"(V_peak={r['V_peak']:.1f} V)  P_loss={r['p_loss_w']:.1f} W")
