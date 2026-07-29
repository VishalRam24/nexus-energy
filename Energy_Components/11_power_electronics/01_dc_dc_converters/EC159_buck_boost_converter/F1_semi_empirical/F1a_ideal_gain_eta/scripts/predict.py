"""EC159 -- Buck-Boost Converter -- F1a Ideal Gain + Efficiency -- Standardized Predict Interface"""
import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from model import BuckBoostConverterF1a


class ComponentModel:
    component_id = "EC159"

    def __init__(self, params_path=None):
        if params_path is None:
            params_path = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = BuckBoostConverterF1a(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        inputs:
            v_in  : float  Input voltage [V]
            D     : float  Duty cycle [0.1 – 0.9]
            P_out : float  Output power [W] (optional, default 0)
        returns:
            D_clamped, voltage_gain, V_out_mag, V_in, eta,
            P_in_W, P_out_W, I_out_A, I_in_A
        """
        v_in  = float(inputs.get("v_in",  24.0))
        D     = float(inputs.get("D",      0.5))
        P_out = float(inputs.get("P_out",  0.0))
        return self._model.predict(v_in, D, P_out)

    def get_info(self) -> dict:
        m = self._model
        return {
            "name":        "Buck-Boost Converter (Inverting)",
            "ec_id":       "EC159",
            "fidelity":    "F1a",
            "model":       "Ideal Gain + Constant Efficiency (V_out = D/(1-D)*V_in, eta=0.88)",
            "description": (
                f"Inverting buck-boost. Ideal gain law: V_out_mag = D/(1-D)*V_in. "
                f"eta={m.eta:.2f} (constant). D range [{m.D_min:.1f}, {m.D_max:.1f}]."
            ),
            "inputs": {
                "v_in":  {"unit": "V",            "range": [1.0, 1000.0]},
                "D":     {"unit": "dimensionless", "range": [0.1, 0.9]},
                "P_out": {"unit": "W",             "range": [0.0, 1e6]},
            },
            "outputs": {
                "D_clamped":    {"unit": "dimensionless"},
                "voltage_gain": {"unit": "dimensionless"},
                "V_out_mag":    {"unit": "V"},
                "V_in":         {"unit": "V"},
                "eta":          {"unit": "dimensionless"},
                "P_in_W":       {"unit": "W"},
                "P_out_W":      {"unit": "W"},
                "I_out_A":      {"unit": "A"},
                "I_in_A":       {"unit": "A"},
            },
            "source": "Erickson & Maksimovic (2001) Fundamentals of Power Electronics, 2nd ed.",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    info = model.get_info()
    print(f"{info['ec_id']} -- {info['name']} -- {info['fidelity']}")
    print("\nVoltage gain vs D:")
    for D in [0.1, 0.3, 0.5, 0.7, 0.9]:
        r = model.predict({"v_in": 24.0, "D": D})
        print(f"  D={D:.1f}: V_out={r['V_out_mag']:.2f} V (gain={r['voltage_gain']:.3f})")
    print("\nWith load (P_out=100W, V_in=24V, D=0.5):")
    r = model.predict({"v_in": 24.0, "D": 0.5, "P_out": 100.0})
    for k, v in r.items():
        print(f"  {k}: {v:.4f}")
