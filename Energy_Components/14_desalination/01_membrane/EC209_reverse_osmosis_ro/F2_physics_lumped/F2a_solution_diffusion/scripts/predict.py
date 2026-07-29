"""
EC209 -- Reverse Osmosis (RO) -- F2a Solution-Diffusion -- predict interface.
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from model import RO_SolutionDiffusion_F2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH) as f:
        return json.load(f)


class ComponentModel:
    component_id = "EC209"
    component_name = "Reverse Osmosis (RO)"
    fidelity = "F2a -- Solution-Diffusion Membrane Model"
    version = "1.0.0"

    def __init__(self, params=None):
        self._raw = _load_params()
        if params:
            self._raw["unit"].update(params)
        self._model = RO_SolutionDiffusion_F2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        inputs:
            feed_concentration_gL : float  Feed NaCl concentration [g/L]  (default 35)
            feed_pressure_bar     : float  Feed pressure [bar]            (default 60)
            feed_flow_m3h         : float  Feed flow rate [m3/h]          (default 8)
            temperature_degC      : float  Feed temperature [degC]        (default 25)
            N_elements            : int    Elements per vessel (default from params)

        returns:
            dict with Qp_m3h, Cp_gL, recovery, rejection, SEC_kwhm3, profiles, ...
        """
        Cf = inputs.get("feed_concentration_gL", 35.0)
        P = inputs.get("feed_pressure_bar", 60.0)
        Qf = inputs.get("feed_flow_m3h", 8.0)
        T = inputs.get("temperature_degC", 25.0)
        N = inputs.get("N_elements", None)

        return self._model.compute(Cf, P, Qf, T, N)

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "feed_concentration_gL": {"unit": "g/L", "range": [1, 45], "default": 35},
                "feed_pressure_bar": {"unit": "bar", "range": [10, 85], "default": 60},
                "feed_flow_m3h": {"unit": "m3/h", "range": [3, 17], "default": 8},
                "temperature_degC": {"unit": "degC", "range": [10, 40], "default": 25},
                "N_elements": {"unit": "-", "range": [1, 8], "default": 7},
            },
            "outputs": {
                "Qp_m3h": "m3/h  (permeate flow)",
                "Cp_gL": "g/L   (permeate concentration)",
                "recovery": "-    (Qp/Qf)",
                "rejection": "-   (1 - Cp/Cf)",
                "SEC_kwhm3": "kWh/m3 (specific energy consumption)",
                "Cc_gL": "g/L   (concentrate concentration)",
                "profiles": "dict  (element-by-element profiles)",
            },
            "source": self._raw.get("unit", {}).get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    r = m.predict({
        "feed_concentration_gL": 35.0,
        "feed_pressure_bar": 60.0,
        "feed_flow_m3h": 8.0,
        "temperature_degC": 25.0,
    })
    print(f"Recovery={r['recovery']:.1%}, Rejection={r['rejection']:.4f}, "
          f"SEC={r['SEC_kwhm3']:.2f} kWh/m3, Cp={r['Cp_gL']:.3f} g/L")
