"""
EC096 -- Magnetic Refrigeration -- F2a AMR (Active Magnetic Regenerator)
Standardised predict() / get_info() interface (mirrors EC001 template).
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import AMR_F2a, GdMagnetocaloric

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for EC096 AMR magnetocaloric refrigerator (F2a)."""

    component_id = "EC096"
    component_name = "Magnetic Refrigeration (Active Magnetic Regenerator)"
    fidelity = "F2a -- AMR cycle, mean-field Gd magnetocaloric material, regenerator ODE"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            self._raw["unit"].update(params)
        self._model = AMR_F2a(self._raw)
        self._mat = self._model.mat

    def predict(self, inputs: dict) -> dict:
        """
        Run AMR cycle to periodic steady state and return cooling performance.

        inputs:
            B_field_max : float [T]   (overrides applied field)
            cycle_freq  : float [Hz]
            mdot_fluid  : float [kg/s]
            T_hot_K     : float [K]
            T_cold_K    : float [K]
            n_cycles    : int (cycles to steady state, default 40)
        """
        u = self._raw["unit"]
        if "B_field_max" in inputs:
            u["B_field_max"]["value"] = float(inputs["B_field_max"])
        if "cycle_freq" in inputs:
            u["cycle_freq"]["value"] = float(inputs["cycle_freq"])
        if "mdot_fluid" in inputs:
            u["mdot_fluid"]["value"] = float(inputs["mdot_fluid"])
        if "T_hot_K" in inputs:
            u["T_hot_K"]["value"] = float(inputs["T_hot_K"])
        if "T_cold_K" in inputs:
            u["T_cold_K"]["value"] = float(inputs["T_cold_K"])

        # rebuild model with any overrides
        self._model = AMR_F2a(self._raw)
        self._mat = self._model.mat
        n_cycles = int(inputs.get("n_cycles", 40))
        return self._model.run_cycle_steady(n_cycles=n_cycles)

    def delta_T_ad(self, T_K: float, B_T: float = None) -> float:
        """Convenience: adiabatic temperature change [K] for field change 0->B."""
        H = (B_T if B_T is not None else self._model.B_max) / (4.0e-7 * 3.141592653589793)
        return self._mat.delta_T_ad(float(T_K), H, 0.0)

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "B_field_max": {"unit": "T", "range": [0.5, 7.0]},
                "cycle_freq": {"unit": "Hz", "range": [0.1, 10.0]},
                "mdot_fluid": {"unit": "kg/s", "range": [0.001, 0.5]},
                "T_hot_K": {"unit": "K", "range": [250, 320]},
                "T_cold_K": {"unit": "K", "range": [240, 315]},
                "n_cycles": {"unit": "-", "range": [5, 200]},
            },
            "outputs": {
                "COP": "-",
                "COP_Carnot": "-",
                "Q_cold_W": "W",
                "Q_hot_W": "W",
                "W_input_W": "W",
                "T_span_K": "K",
                "T_solid_profile": "K array",
                "dTad_profile": "K array",
                "energy_residual_W": "W",
            },
            "source": self._raw.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    dTad = m.delta_T_ad(293.0, 1.5)
    print(f"Gd Delta T_ad at T_C (293 K, 1.5 T) = {dTad:.2f} K (expect ~4-5 K)")
    r = m.predict({"n_cycles": 30})
    print(f"COP={r['COP']:.3f}  COP_Carnot={r['COP_Carnot']:.2f}  "
          f"Q_cold={r['Q_cold_W']:.2f} W  W_in={r['W_input_W']:.2f} W  "
          f"span={r['T_span_K']:.1f} K")
