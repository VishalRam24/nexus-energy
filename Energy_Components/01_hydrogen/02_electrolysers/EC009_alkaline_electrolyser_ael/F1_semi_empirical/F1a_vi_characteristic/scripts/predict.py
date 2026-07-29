"""EC009 — Alkaline Electrolyser (AEL) — F1a V-I Characteristic — Standardized Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import AELF1a


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = AELF1a(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        Predict AEL V-I characteristic outputs.

        Args:
            inputs: dict with keys:
                - current_density (A/m2): scalar or array
                - temperature     (degC): scalar or array, default 80

        Returns:
            dict with keys:
                - cell_voltage      (V)
                - stack_voltage     (V)
                - hydrogen_rate_mols (mol/s)
                - power_kw          (kW)
                - efficiency        (-)
        """
        j = np.asarray(inputs["current_density"], dtype=float)
        T_C = np.asarray(inputs.get("temperature", 80.0), dtype=float)
        T_K = T_C + 273.15

        return {
            "cell_voltage":        self._model.cell_voltage(j, T_K),
            "stack_voltage":       self._model.stack_voltage(j, T_K),
            "hydrogen_rate_mols":  self._model.hydrogen_rate(j, T_K),
            "power_kw":            self._model.power_kw(j, T_K),
            "efficiency":          self._model.efficiency(j, T_K),
        }

    def get_info(self) -> dict:
        return {
            "name": "Alkaline Electrolyser (AEL)",
            "ec_id": "EC009",
            "fidelity": "F1a",
            "description": "Ulleberg (2003) V-I characteristic: V = E_rev(T) + r(T)/A*j + s*log10((t1+t2/T+t3/T^2)*j/A + 1)",
            "inputs": {
                "current_density": {"unit": "A/m2", "range": [0.0, 3000.0]},
                "temperature":     {"unit": "degC", "range": [40.0, 90.0], "default": 80.0},
            },
            "outputs": {
                "cell_voltage":       {"unit": "V"},
                "stack_voltage":      {"unit": "V"},
                "hydrogen_rate_mols": {"unit": "mol/s"},
                "power_kw":           {"unit": "kW"},
                "efficiency":         {"unit": "dimensionless"},
            },
            "source": "Ulleberg (2003), Int. J. Hydrogen Energy, 28(1), 21-33",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    r = model.predict({"current_density": 2000.0, "temperature": 80.0})
    print(f"At j=2000 A/m2, T=80C:")
    print(f"  V_cell={float(r['cell_voltage']):.3f} V")
    print(f"  V_stack={float(r['stack_voltage']):.2f} V")
    print(f"  H2={float(r['hydrogen_rate_mols'])*3600:.4f} mol/hr")
    print(f"  P={float(r['power_kw']):.3f} kW")
    print(f"  eta={float(r['efficiency'])*100:.1f} %")
