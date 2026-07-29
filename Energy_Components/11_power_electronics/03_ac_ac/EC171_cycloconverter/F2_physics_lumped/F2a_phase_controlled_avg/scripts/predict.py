"""
EC171 -- Cycloconverter -- F2a Physics-Lumped Phase-Controlled Averaged Model
Standardised predict() / get_info() interface (mirrors EC001 F2a).
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import CycloconverterF2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for the EC171 cycloconverter F2a physics-lumped model."""

    component_id = "EC171"
    component_name = "Cycloconverter"
    fidelity = "F2a -- Phase-Controlled Averaged Model with R-L Output-Current ODE"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            self._raw["unit"].update(params)
        self._model = CycloconverterF2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        Run the lumped averaged-converter + R-L load simulation.

        inputs:
            r_mod : float       output voltage modulation ratio in [0,1] (default 0.8)
            f_out : float       output frequency [Hz] (default 10.0, must be < f_line)
            n_cycles : int      output cycles to simulate (default 4)
            n_pts_per_cycle: int integration sample density (default 400)
            phi_load : float    optional load displacement angle override [rad]
        """
        r_mod = inputs.get("r_mod", 0.8)
        f_out = inputs.get("f_out", 10.0)
        n_cycles = inputs.get("n_cycles", 4)
        n_pts = inputs.get("n_pts_per_cycle", 400)
        phi_load = inputs.get("phi_load", None)

        return self._model.simulate(r_mod, f_out, n_cycles, n_pts, phi_load)

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "r_mod": {"unit": "-", "range": [0.0, 1.0]},
                "f_out": {"unit": "Hz", "range": [0.5, 16.0],
                          "note": "must be < f_line; design rule f_out < f_line/3"},
                "n_cycles": {"unit": "-"},
                "n_pts_per_cycle": {"unit": "-"},
                "phi_load": {"unit": "rad"},
            },
            "outputs": {
                "t": "s",
                "i_out": "A",
                "v_out_avg": "V",
                "alpha": "rad",
                "I_out_rms": "A",
                "V_out_ll_rms": "V",
                "P_out_total": "W",
                "P_loss_total": "W",
                "P_in_total": "W",
                "efficiency": "-",
                "input_displacement_factor": "- (lagging)",
                "output_thd": "-",
                "dominant_harmonics_hz": "Hz",
            },
            "source": self._raw.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    r = m.predict({"r_mod": 0.8, "f_out": 10.0, "n_cycles": 4})
    print(
        f"f_out={r['f_out']} Hz (< f_line {r['f_line']} Hz, ratio={r['freq_ratio']:.3f}), "
        f"V_out_ll_rms={r['V_out_ll_rms']:.1f} V, I_rms={r['I_out_rms']:.1f} A, "
        f"P_out={r['P_out_total']/1e3:.1f} kW, eta={r['efficiency']:.4f}, "
        f"DPF_in={r['input_displacement_factor']:.3f} (lagging), THD={r['output_thd']:.3f}"
    )
