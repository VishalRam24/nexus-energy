"""
EC163 -- Single-Phase DC-AC Inverter -- F2a Averaged SPWM + LC Filter
Standardised predict() / get_info() interface (mirrors EC001 template).
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import SinglePhaseInverterF2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for EC163 F2a averaged SPWM inverter model."""

    component_id = "EC163"
    component_name = "Single-Phase DC-AC Inverter"
    fidelity = "F2a -- Averaged SPWM H-Bridge with LC Filter ODE"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            self._raw["unit"].update(params)
        self._model = SinglePhaseInverterF2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        Run averaged SPWM + LC-filter dynamic simulation at one operating point.

        inputs:
            V_dc : float        (DC bus voltage, default from params)
            m_a : float         (modulation index 0..1, default 0.85)
            f_grid : float      (output fundamental frequency, default 50 Hz)
            R_load : float      (resistive load ohms, default from params)
            duration_s : float  (sim length, default ~6 fundamental cycles)
            dt : float          (sample step, default T_grid/400)
        """
        V_dc = inputs.get("V_dc", None)
        m_a = inputs.get("m_a", 0.85)
        f_grid = inputs.get("f_grid", None)
        R_load = inputs.get("R_load", None)
        duration_s = inputs.get("duration_s", None)
        dt = inputs.get("dt", None)

        return self._model.operating_point(
            v_dc=V_dc, m_a=m_a, f_grid=f_grid, R_load=R_load,
            duration_s=duration_s, dt=dt,
        )

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "V_dc": {"unit": "V", "range": [200, 800]},
                "m_a": {"unit": "-", "range": [0.0, 1.0]},
                "f_grid": {"unit": "Hz", "range": [40, 60]},
                "R_load": {"unit": "Ohm", "range": [1, 1000]},
                "duration_s": {"unit": "s"},
                "dt": {"unit": "s"},
            },
            "outputs": {
                "t": "s",
                "v_inv": "V (pre-filter bridge voltage)",
                "i_L": "A (filter inductor current)",
                "v_out": "V (filtered output voltage)",
                "v_out_rms": "V",
                "i_out_rms": "A",
                "p_out_w": "W",
                "p_conduction_w": "W",
                "p_switching_w": "W",
                "p_loss_total_w": "W",
                "efficiency": "-",
                "thd_prefilter": "-",
                "thd_postfilter": "-",
                "f_lc_hz": "Hz",
            },
            "source": self._raw.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    r = m.predict({"m_a": 0.85, "duration_s": 0.12})
    print(f"V_out_rms: {r['v_out_rms']:.2f} V (ideal fund {r['v_fund_rms_ideal']:.2f} V), "
          f"P_out: {r['p_out_w']:.1f} W, eta: {r['efficiency']*100:.2f}%, "
          f"THD_post: {r['thd_postfilter']*100:.3f}%, f_LC: {r['f_lc_hz']:.0f} Hz")
