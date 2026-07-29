"""EC036 — VRFB — F1a Nernst+Ohmic — Standardized Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import VRFBF1a


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = VRFBF1a(self.params)

    def predict(self, inputs: dict) -> dict:
        soc     = np.asarray(inputs["soc"],     dtype=float)
        current = np.asarray(inputs["current"], dtype=float)

        # Validate SOC range
        if np.any(soc < 0.0) or np.any(soc > 1.0):
            raise ValueError(
                f"SOC must be in [0, 1]. Got min={float(np.min(soc)):.6g}, "
                f"max={float(np.max(soc)):.6g}."
            )
        # Clamp to physically safe operating window
        soc = np.clip(soc, VRFBF1a.SOC_MIN, VRFBF1a.SOC_MAX)

        return {
            "cell_voltage":   self._model.cell_voltage(soc, current),
            "stack_voltage":  self._model.stack_voltage(soc, current),
            "power":          self._model.power_w(soc, current),
            "efficiency":     self._model.efficiency(soc, current),
        }

    def get_info(self) -> dict:
        return {
            "name":        "Vanadium Redox Flow Battery (VRFB)",
            "ec_id":       "EC036",
            "fidelity":    "F1a",
            "description": "E_Nernst(SOC) = E0 + 2*(RT/nF)*ln(SOC/(1-SOC)); V_cell = E_Nernst - I*R_cell; V_stack = N_cells*V_cell",
            "inputs": {
                "soc":     {"unit": "dimensionless", "range": [0.1, 0.9]},
                "current": {"unit": "A", "range": [-100.0, 100.0], "note": "positive=discharge"},
            },
            "outputs": {
                "cell_voltage":  {"unit": "V"},
                "stack_voltage": {"unit": "V"},
                "power":         {"unit": "W", "note": "positive=discharge"},
                "efficiency":    {"unit": "dimensionless", "note": "voltage efficiency, valid for I>0"},
            },
            "source":  "Blanc & Rufer (2010), Paths to Sustainable Energy",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    r = model.predict({"soc": 0.5, "current": 50.0})
    print(f"SOC=50%, 50A discharge: "
          f"V_cell={float(r['cell_voltage']):.4f} V, "
          f"V_stack={float(r['stack_voltage']):.2f} V, "
          f"P={float(r['power'])/1000:.2f} kW, "
          f"eta={float(r['efficiency']):.3f}")
