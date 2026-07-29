"""
EC042 -- Pseudocapacitor -- F1b RC-Thermal -- Standardized Predict Interface

Usage:
    model = ComponentModel()
    result = model.predict({"v_cap": 0.5, "current": 50.0, "temperature": 298.15})
"""

import json
import numpy as np
from pathlib import Path
from model import PseudocapacitorF1b


class ComponentModel:
    """Standardized interface for EC042 Pseudocapacitor -- F1b RC-thermal model."""

    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = PseudocapacitorF1b(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        Args:
            inputs: {
                "v_cap": float or array in V (capacitor voltage, 0 to 1.0 V),
                "current": float or array in A (positive=discharge),
                "temperature": float or array in K (default 298.15; range 243-333 K)
            }
        Returns: {
            "terminal_voltage": V,
            "power": W,
            "heat_generation": W,
            "esr": Ohm,
            "capacitance": F,
            "soc": dimensionless (V_cap/V_max),
            "stored_energy": J,
            "leakage_current": A,
            "dvcap_dt": V/s
        }
        """
        v_cap = np.asarray(inputs["v_cap"], dtype=float)
        current = np.asarray(inputs["current"], dtype=float)
        temperature = np.asarray(inputs.get("temperature", 298.15), dtype=float)

        return {
            "terminal_voltage": self._model.terminal_voltage(v_cap, current, temperature),
            "power": self._model.power(v_cap, current, temperature),
            "heat_generation": self._model.heat_generation(v_cap, current, temperature),
            "esr": self._model.esr(temperature),
            "capacitance": self._model.capacitance(temperature),
            "soc": self._model.soc(v_cap),
            "stored_energy": self._model.stored_energy(v_cap, temperature),
            "leakage_current": self._model.leakage_current(v_cap),
            "dvcap_dt": self._model.vcap_derivative(v_cap, current, temperature),
        }

    def get_info(self) -> dict:
        return {
            "name": "Pseudocapacitor (RuO2 representative)",
            "ec_id": "EC042",
            "fidelity": "F1b",
            "description": "RC thermal model; Arrhenius ESR(T) with E_a=12 kJ/mol, linear C(T), faradaic entropic heat, leakage; -30 to 60 degC",
            "inputs": {
                "v_cap": {"unit": "V", "range": [0.0, 1.0],
                          "note": "Capacitor voltage (state variable); RuO2 window 0-1 V in H2SO4"},
                "current": {"unit": "A", "range": [-100.0, 100.0],
                            "note": "positive=discharge, negative=charge"},
                "temperature": {"unit": "K", "range": [243.15, 333.15],
                                "note": "-30 to 60 degC (aqueous electrolyte)"},
            },
            "outputs": {
                "terminal_voltage": {"unit": "V"},
                "power": {"unit": "W"},
                "heat_generation": {"unit": "W", "note": "Joule + faradaic entropic term"},
                "esr": {"unit": "Ohm"},
                "capacitance": {"unit": "F"},
                "soc": {"unit": "dimensionless", "note": "V_cap / V_max"},
                "stored_energy": {"unit": "J"},
                "leakage_current": {"unit": "A"},
                "dvcap_dt": {"unit": "V/s"},
            },
            "source": "Conway (1999); Trasatti & Buzzanca (1971); Zheng et al. (1995); Sugimoto et al. (2006)",
        }


if __name__ == "__main__":
    model = ComponentModel()
    print(json.dumps(model.get_info(), indent=2))
    result = model.predict({"v_cap": 0.5, "current": 50.0, "temperature": 298.15})
    print(f"\nAt V_cap=0.5V (50% SOC), I=50A, T=298.15K:")
    for k, v in result.items():
        val = float(v) if np.ndim(v) == 0 else v
        print(f"  {k}: {val:.6g}")
