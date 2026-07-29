"""
EC062 -- HAWT Onshore -- F2a BEM Steady -- predict interface.
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from model import HAWT_BEM_F2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")

def _load_params():
    with open(_PARAMS_PATH) as f:
        return json.load(f)

class ComponentModel:
    component_id = "EC062"
    component_name = "HAWT Onshore Wind Turbine"
    fidelity = "F2a -- BEM Steady-State"
    version = "1.0.0"

    def __init__(self, params=None):
        self._raw = _load_params()
        if params:
            self._raw["turbine"].update(params)
        self._model = HAWT_BEM_F2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        V = inputs.get("wind_speed_m_s", 10.0)
        pitch = inputs.get("pitch_deg", 0.0)
        rpm = inputs.get("rpm", None)
        tsr = inputs.get("tip_speed_ratio", None)
        return self._model.solve(V, pitch, rpm, tsr)

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity, "version": self.version,
            "inputs": {
                "wind_speed_m_s": {"unit": "m/s", "range": [3, 25]},
                "pitch_deg": {"unit": "deg"}, "rpm": {"unit": "rpm"},
                "tip_speed_ratio": {"unit": "-"},
            },
            "outputs": {
                "power_kw": "kW", "thrust_kN": "kN", "Cp": "-", "Ct": "-",
                "blade_loads": "list of dicts per element",
            },
            "source": self._raw.get("source", ""),
        }

if __name__ == "__main__":
    m = ComponentModel()
    r = m.predict({"wind_speed_m_s": 10.0})
    print(f"P={r['power_kw']:.1f} kW, Cp={r['Cp']:.3f}, Ct={r['Ct']:.3f}")
