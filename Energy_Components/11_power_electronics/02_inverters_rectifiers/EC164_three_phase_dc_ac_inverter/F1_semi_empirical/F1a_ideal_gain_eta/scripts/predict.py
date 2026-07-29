"""
EC164 — Three-Phase DC-AC Inverter — F1a Ideal Gain + Efficiency
ComponentModel wrapper with standardised predict() / get_info() interface.
"""

import json
import os
from model import ThreePhaseInverterModel

_PARAMS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "parameters.json"
)


def _load_default_params() -> dict:
    with open(_PARAMS_PATH, "r") as fh:
        return json.load(fh)["default_parameters"]


class ComponentModel:
    """
    Standardised wrapper for the Three-Phase DC-AC Inverter model.

    Usage
    -----
    >>> model = ComponentModel()
    >>> out = model.predict({"v_dc": 800.0, "p_load": 80000.0, "modulation_index": 0.9})
    >>> print(out)
    """

    component_id   = "EC164"
    component_name = "Three-Phase DC-AC Inverter"
    fidelity       = "F1a — Ideal Gain + Part-Load Efficiency"
    version        = "1.0.0"

    def __init__(self, params: dict = None):
        defaults = _load_default_params()
        if params:
            defaults.update(params)
        self._params  = defaults
        self._physics = ThreePhaseInverterModel(defaults)

    # ------------------------------------------------------------------
    # Primary interface
    # ------------------------------------------------------------------

    def predict(self, inputs: dict) -> dict:
        """
        Predict inverter operating point.

        Parameters
        ----------
        inputs : dict
            v_dc             : float — DC bus voltage [V]  (> 0)
            p_load           : float — output power [W]    (0 – P_rated)
            modulation_index : float — SVPWM modulation index [0, 1]
            power_factor     : float — load power factor [-], optional (default 1.0)

        Returns
        -------
        dict
            v_ac_rms_V, i_ac_rms_A, efficiency, p_in_W, p_loss_W, PLR
        """
        v_dc  = float(inputs["v_dc"])
        p_load = float(inputs["p_load"])
        m     = float(inputs["modulation_index"])
        pf    = float(inputs.get("power_factor", 1.0))

        if v_dc <= 0:
            raise ValueError(f"v_dc must be > 0, got {v_dc}")
        if not (0.0 <= p_load <= self._params["P_rated"]):
            raise ValueError(
                f"p_load must be in [0, {self._params['P_rated']}] W, got {p_load}"
            )
        if not (0.0 <= m <= 1.0):
            raise ValueError(f"modulation_index must be in [0, 1], got {m}")

        result = self._physics.evaluate(v_dc, p_load, m, pf)

        return {
            "v_ac_rms_V":   round(result["v_ac_rms_V"],  4),
            "i_ac_rms_A":   round(result["i_ac_rms_A"],  4),
            "efficiency":   round(result["efficiency"],   6),
            "p_in_W":       round(result["p_in_W"],       2),
            "p_loss_W":     round(result["p_loss_W"],     2),
            "PLR":          round(result["PLR"],          4),
        }

    def get_info(self) -> dict:
        """Return component metadata and parameter summary."""
        return {
            "component_id":   self.component_id,
            "component_name": self.component_name,
            "fidelity":       self.fidelity,
            "version":        self.version,
            "inputs": {
                "v_dc":             {"unit": "V",  "range": [0, None]},
                "p_load":           {"unit": "W",  "range": [0, self._params["P_rated"]]},
                "modulation_index": {"unit": "-",  "range": [0.0, 1.0]},
                "power_factor":     {"unit": "-",  "range": [0.0, 1.0]},
            },
            "outputs": {
                "v_ac_rms_V":  "V (line-to-line RMS)",
                "i_ac_rms_A":  "A (line RMS)",
                "efficiency":  "-",
                "p_in_W":      "W",
                "p_loss_W":    "W",
                "PLR":         "-",
            },
            "active_parameters": self._params,
            "source": (
                "Mohan, Undeland & Robbins (2003), 'Power Electronics,' 3rd ed. Wiley. "
                "Part-load efficiency: IEC 61683 / EN 50530."
            ),
        }


if __name__ == "__main__":
    model = ComponentModel()
    print(model.get_info())
    print(model.predict({"v_dc": 800.0, "p_load": 80000.0, "modulation_index": 0.9}))
