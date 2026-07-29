"""EC010 — Solid Oxide Electrolyser (SOEC) — F1a V-I Characteristic — Standardized Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import SOECF1a


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = SOECF1a(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        Predict SOEC V-I characteristic outputs.

        Args:
            inputs: dict with keys:
                - current_density (A/cm2): scalar or array
                - temperature     (degC):  scalar or array, default 800

        Returns:
            dict with keys:
                - cell_voltage       (V)
                - stack_voltage      (V)
                - hydrogen_rate_mols (mol/s)
                - power_kw           (kW)
                - efficiency         (-)
                - asr                (Ohm.cm2)
        """
        j = np.asarray(inputs["current_density"], dtype=float)
        T_C = np.asarray(inputs.get("temperature", 800.0), dtype=float)
        T_K = T_C + 273.15

        return {
            "cell_voltage":        self._model.cell_voltage(j, T_K),
            "stack_voltage":       self._model.stack_voltage(j, T_K),
            "hydrogen_rate_mols":  self._model.hydrogen_rate(j),
            "power_kw":            self._model.power_kw(j, T_K),
            "efficiency":          self._model.efficiency(j, T_K),
            "asr":                 self._model.asr(T_K),
        }

    def get_info(self) -> dict:
        return {
            "name": "Solid Oxide Electrolyser (SOEC)",
            "ec_id": "EC010",
            "fidelity": "F1a",
            "description": "ASR-based V-I model: V = E_rev(T) + j*ASR(T), ASR Arrhenius-type",
            "inputs": {
                "current_density": {"unit": "A/cm2", "range": [0.0, 2.0]},
                "temperature":     {"unit": "degC",  "range": [600.0, 900.0], "default": 800.0},
            },
            "outputs": {
                "cell_voltage":       {"unit": "V"},
                "stack_voltage":      {"unit": "V"},
                "hydrogen_rate_mols": {"unit": "mol/s"},
                "power_kw":           {"unit": "kW"},
                "efficiency":         {"unit": "dimensionless"},
                "asr":                {"unit": "Ohm.cm2"},
            },
            "source": "Ni et al. (2007), Chem. Eng. Tech., 29(6), 636-642",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    r = model.predict({"current_density": 1.0, "temperature": 800.0})
    print(f"At j=1.0 A/cm2, T=800C:")
    print(f"  V_cell={float(r['cell_voltage']):.3f} V")
    print(f"  V_stack={float(r['stack_voltage']):.2f} V")
    print(f"  H2={float(r['hydrogen_rate_mols'])*3600:.4f} mol/hr")
    print(f"  P={float(r['power_kw']):.3f} kW")
    print(f"  eta={float(r['efficiency'])*100:.1f} %")
    print(f"  ASR={float(r['asr']):.3f} Ohm.cm2")
