"""
EC174 -- Instrument Transformer (CT / PT) -- F2a Magnetizing-Branch
Standardised predict() / get_info() interface (mirrors EC001 template).
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import InstrumentTransformerF2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for EC174 instrument-transformer F2a model."""

    component_id = "EC174"
    component_name = "Instrument Transformer (CT / PT)"
    fidelity = "F2a -- Magnetizing-Branch Equivalent Circuit with Nonlinear-Flux ODE"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            # Each unit field is wrapped as {"value": ..., "unit": ..., "note": ...}.
            # Merge overrides into the existing wrapper (updating only "value")
            # rather than replacing the dict with a bare scalar/string, which would
            # break downstream u[key]["value"] indexing.
            unit = self._raw["unit"]
            for key, val in params.items():
                if isinstance(val, dict):
                    unit.setdefault(key, {}).update(val)
                elif isinstance(unit.get(key), dict):
                    unit[key]["value"] = val
                else:
                    unit[key] = {"value": val}
        self._model = InstrumentTransformerF2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        inputs:
            mode             : "accuracy" (default) or "saturation"
            i_primary_A      : float/list, primary current (CT)
            v_primary_V      : float/list, primary voltage (PT)
            burden_fraction  : float, burden as multiple of rated VA (default 1.0)
            i_primary_peak_A : float, peak primary current for saturation sim
            n_cycles         : float, cycles to simulate (saturation, default 2)

        returns (accuracy mode):
            ratio_error_pct, phase_error_min, phase_error_deg, within_class, ...
        returns (saturation mode):
            t, i_sec, i_sec_ideal, i_mag, flux, distortion, saturated, energy_resid
        """
        mode = inputs.get("mode", "accuracy")
        burden = inputs.get("burden_fraction", 1.0)

        if mode == "saturation":
            i_peak = inputs.get("i_primary_peak_A",
                                self._model.I_rated * np.sqrt(2.0) * 10.0)
            n_cycles = inputs.get("n_cycles", 2.0)
            res = self._model.simulate_saturation(i_peak, burden, n_cycles=n_cycles)
            return res

        # accuracy mode
        if self._model.type == "CT":
            inp = inputs.get("i_primary_A", self._model.I_rated)
        else:
            inp = inputs.get("v_primary_V", self._model.V_rated)
        res = self._model.accuracy(inp, burden)
        res = {k: (np.asarray(v).tolist() if np.ndim(v) else float(v))
               for k, v in res.items()}
        res["within_class"] = self._model.within_accuracy_class(inp, burden)
        res["accuracy_class"] = self._model.accuracy_class
        return res

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "type": self._model.type,
            "inputs": {
                "mode": {"options": ["accuracy", "saturation"]},
                "i_primary_A": {"unit": "A", "range": [0, 4000]},
                "v_primary_V": {"unit": "V", "range": [0, 15000]},
                "burden_fraction": {"unit": "-", "range": [0, 4]},
                "i_primary_peak_A": {"unit": "A"},
                "n_cycles": {"unit": "-"},
            },
            "outputs": {
                "ratio_error_pct": "%",
                "phase_error_min": "minutes",
                "phase_error_deg": "deg",
                "within_class": "bool",
                "t": "s", "i_sec": "A", "i_mag": "A", "flux": "Wb-turns",
                "distortion": "-", "saturated": "bool", "energy_resid": "-",
            },
            "source": self._raw.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    r = m.predict({"mode": "accuracy", "i_primary_A": 200.0, "burden_fraction": 1.0})
    print(f"Accuracy @rated: ratio_err={r['ratio_error_pct']:.4f} %, "
          f"phase={r['phase_error_min']:.2f} min, within_class={r['within_class']}")
    s = m.predict({"mode": "saturation", "i_primary_peak_A": 200*1.414*15, "n_cycles": 2})
    print(f"Saturation @15x: distortion={s['distortion']:.3f}, saturated={s['saturated']}, "
          f"energy_resid={s['energy_resid']:.2e}")
