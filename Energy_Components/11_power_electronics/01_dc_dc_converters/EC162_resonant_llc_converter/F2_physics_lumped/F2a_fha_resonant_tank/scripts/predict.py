"""
EC162 -- Resonant LLC Converter -- F2a Physics-Lumped (FHA)
Standardised predict() / get_info() interface (mirrors EC001 template).
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import LLCConverterF2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for LLC F2a FHA physics-lumped model."""

    component_id = "EC162"
    component_name = "Resonant LLC Converter"
    fidelity = "F2a -- FHA Resonant Tank with Output-Filter Transient ODE"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            self._raw["unit"].update(params)
        self._model = LLCConverterF2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        Run the lumped output-filter transient and report the steady-state
        operating point.

        inputs:
            f_sw_Hz   : float  switching frequency (default = f_r)
            r_load    : float  output load resistance [Ohm] (default sized for rated power)
            v_in      : float  input DC bus voltage [V] (default nominal)
            v_out0    : float  initial output voltage [V] (default 0)
            dt        : float  ODE output step [s] (default 1e-6)
            duration_s: float  transient duration [s] (default 2e-3)
        """
        m = self._model
        f_sw = inputs.get("f_sw_Hz", m.f_r)
        fn = f_sw / m.f_r
        # default load: deliver rated power at nominal output
        r_default = m.V_out_nom ** 2 / m.P_rated
        r_load = inputs.get("r_load", r_default)
        v_in = inputs.get("v_in", m.V_in_nom)
        v_out0 = inputs.get("v_out0", 0.0)
        dt = inputs.get("dt", 1.0e-6)
        dur = inputs.get("duration_s", 2.0e-3)

        sim = m.simulate(fn, r_load, v_in, v_out0, dt, dur)
        op = m.operating_point(fn, r_load, v_in)
        losses = m.loss_breakdown(fn, r_load, v_in)
        out = dict(sim)
        out["fn"] = fn
        out["f_sw_Hz"] = f_sw
        out["operating_point"] = op
        out["losses"] = losses
        return out

    def get_info(self) -> dict:
        m = self._model
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "derived": {
                "f_r_Hz": m.f_r,
                "Z_0_ohm": m.Z_0,
                "k_ratio": m.k,
            },
            "inputs": {
                "f_sw_Hz": {"unit": "Hz", "range": [50000.0, 300000.0]},
                "r_load": {"unit": "Ohm", "range": [0.05, 20.0]},
                "v_in": {"unit": "V", "range": [200.0, 600.0]},
                "v_out0": {"unit": "V"},
                "dt": {"unit": "s"},
                "duration_s": {"unit": "s"},
            },
            "outputs": {
                "t": "s",
                "v_out": "V (transient)",
                "i_load": "A",
                "p_out": "W",
                "v_out_ss": "V",
                "gain": "- (M)",
                "efficiency": "-",
                "zvs": "bool",
                "operating_point": "dict",
                "losses": "dict (W)",
            },
            "source": self._raw.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    r = m.predict({"duration_s": 2.0e-3, "dt": 2.0e-6})
    print(
        f"f_r = {m._model.f_r/1e3:.1f} kHz | gain M = {r['gain']:.4f} | "
        f"V_out_ss = {r['v_out_ss']:.3f} V | V_out(final) = {r['v_out'][-1]:.3f} V | "
        f"eta = {r['efficiency']*100:.2f}% | ZVS = {r['zvs']}"
    )
