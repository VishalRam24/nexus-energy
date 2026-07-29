"""
EC008 -- PEMEL -- F2a Electrochemical -- predict interface.
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from model import PEMEL_F2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")

def _load_params():
    with open(_PARAMS_PATH) as f:
        return json.load(f)

class ComponentModel:
    component_id = "EC008"
    component_name = "PEM Electrolyser (PEMEL)"
    fidelity = "F2a -- Full Electrochemical + Thermal ODE"
    version = "1.0.0"

    def __init__(self, params=None):
        self._raw = _load_params()
        if params:
            self._raw["unit"].update(params)
        self._model = PEMEL_F2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        j = inputs.get("current_density", 1.0)
        T0 = inputs.get("T_K", 353.15)
        P = inputs.get("P_bar", 30.0)
        dt = inputs.get("dt", 0.1)
        dur = inputs.get("duration_s", 60.0)
        return self._model.simulate(j, T0, P, dt, dur)

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity, "version": self.version,
            "inputs": {
                "current_density": {"unit": "A/cm2", "range": [0, 3]},
                "T_K": {"unit": "K"}, "P_bar": {"unit": "bar"},
                "dt": {"unit": "s"}, "duration_s": {"unit": "s"},
            },
            "outputs": {
                "t": "s", "voltage": "V", "h2_production_kg_s": "kg/s",
                "efficiency": "-", "heat_generation_W": "W", "temperature": "K",
            },
            "source": self._raw.get("source", ""),
        }

if __name__ == "__main__":
    m = ComponentModel()
    r = m.predict({"current_density": 1.0, "duration_s": 10, "dt": 1})
    print(f"V={r['voltage'][-1]:.4f}, H2={r['h2_production_kg_s'][-1]:.6f} kg/s")
