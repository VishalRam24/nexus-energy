"""
EC044 -- Mono-Si PV -- F2a Diode Shading -- predict interface.
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from model import PV_DiodeShading_F2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")

def _load_params():
    with open(_PARAMS_PATH) as f:
        return json.load(f)

class ComponentModel:
    component_id = "EC044"
    component_name = "Monocrystalline Silicon PV"
    fidelity = "F2a -- Single-Diode + Bypass Diode + Partial Shading"
    version = "1.0.0"

    def __init__(self, params=None):
        self._raw = _load_params()
        if params:
            self._raw["module"].update(params)
        self._model = PV_DiodeShading_F2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        inputs:
            irradiance_per_cell : list/array of N_s irradiance values [W/m2]
                                  OR float (uniform irradiance)
            temperature_degC : float (default 25)
        """
        G = inputs.get("irradiance_per_cell", 1000.0)
        T = inputs.get("temperature_degC", 25.0)
        N_s = self._model.N_s

        if isinstance(G, (int, float)):
            G = np.full(N_s, float(G))
        G = np.asarray(G)

        return self._model.iv_curve(G, T)

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity, "version": self.version,
            "inputs": {
                "irradiance_per_cell": {"unit": "W/m2", "note": "array of N_s values or scalar"},
                "temperature_degC": {"unit": "degC", "range": [-20, 80]},
            },
            "outputs": {
                "I": "A", "V": "V", "P": "W",
                "P_mp": "W", "V_mp": "V", "I_mp": "A",
                "num_local_maxima": "-", "shading_loss_pct": "%",
            },
            "source": self._raw.get("source", ""),
        }

if __name__ == "__main__":
    m = ComponentModel()
    r = m.predict({"irradiance_per_cell": 1000.0, "temperature_degC": 25.0})
    print(f"P_mp={r['P_mp']:.1f} W, V_mp={r['V_mp']:.1f} V, I_mp={r['I_mp']:.2f} A")
