"""EC011 — AEM Electrolyser — F1a V-I Curve — Standardized Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import AEMF1a


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = AEMF1a(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        Predict AEM electrolyser V-I outputs.

        Args:
            inputs: dict with keys:
                - current_density (A/m2): scalar or array
                - temperature     (degC): scalar or array, default 60

        Returns:
            dict with keys:
                - cell_voltage      (V)
                - stack_voltage     (V)
                - hydrogen_rate_mols (mol/s)
                - power_kw          (kW)
                - efficiency        (-)
        """
        j = np.asarray(inputs["current_density"], dtype=float)
        T_C = np.asarray(inputs.get("temperature", 60.0), dtype=float)
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
            "name": "Anion Exchange Membrane Electrolyser (AEM)",
            "ec_id": "EC011",
            "fidelity": "F1a",
            "description": "Tafel + ohmic V-I polarization: V = E_rev(T) + Tafel(a) + Tafel(c) + ASR(T)*j",
            "inputs": {
                "current_density": {"unit": "A/m2", "range": [0.0, 20000.0]},
                "temperature":     {"unit": "degC", "range": [30.0, 80.0], "default": 60.0},
            },
            "outputs": {
                "cell_voltage":       {"unit": "V"},
                "stack_voltage":      {"unit": "V"},
                "hydrogen_rate_mols": {"unit": "mol/s"},
                "power_kw":           {"unit": "kW"},
                "efficiency":         {"unit": "dimensionless"},
            },
            "source": "Vincent & Bessarabov (2018); Henkensmeier et al. (2021)",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    r = model.predict({"current_density": 5000.0, "temperature": 60.0})
    print("At j=5000 A/m2 (0.5 A/cm2), T=60C:")
    print(f"  V_cell={float(r['cell_voltage']):.3f} V")
    print(f"  V_stack={float(r['stack_voltage']):.2f} V")
    print(f"  H2={float(r['hydrogen_rate_mols'])*3600:.4f} mol/hr")
    print(f"  P={float(r['power_kw']):.3f} kW")
    print(f"  eta={float(r['efficiency'])*100:.1f} %")
