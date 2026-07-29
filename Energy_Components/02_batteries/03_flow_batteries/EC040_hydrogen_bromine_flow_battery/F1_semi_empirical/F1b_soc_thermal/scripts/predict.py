"""
EC040 -- Hydrogen-Bromine Flow Battery -- F1b SOC-Thermal -- Standardized Predict Interface

Usage:
    model = ComponentModel()
    result = model.predict({"soc": 0.5, "current": 80.0, "temperature": 313.15})
"""

import json
import numpy as np
from pathlib import Path
from model import HydrogenBromineFlowF1b


class ComponentModel:
    """Standardized interface for EC040 Hydrogen-Bromine Flow Battery -- F1b SOC-thermal model."""

    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = HydrogenBromineFlowF1b(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        Args:
            inputs: {
                "soc": float or array (0-1, internally clamped 0.01-0.99),
                "current": float or array in A (positive=discharge),
                "temperature": float or array in K (default 298.15; range 293-333 K)
            }
        Returns: {
            "stack_voltage": V,
            "cell_voltage": V,
            "power": W (net: electrical minus pump losses),
            "heat_generation": W,
            "pump_loss": W,
            "internal_resistance_cell": Ohm (per cell),
            "e_nernst": V (per cell),
            "efficiency": dimensionless
        }
        """
        soc = np.asarray(inputs["soc"], dtype=float)
        current = np.asarray(inputs["current"], dtype=float)
        temperature = np.asarray(inputs.get("temperature", 298.15), dtype=float)

        return {
            "stack_voltage": self._model.stack_voltage(soc, current, temperature),
            "cell_voltage": self._model.cell_voltage(soc, current, temperature),
            "power": self._model.power_w(soc, current, temperature),
            "heat_generation": self._model.heat_generation(soc, current, temperature),
            "pump_loss": self._model.pump_loss(current),
            "internal_resistance_cell": self._model.r_cell(temperature),
            "e_nernst": self._model.e_nernst(soc, temperature),
            "efficiency": self._model.efficiency(soc, current, temperature),
        }

    def get_info(self) -> dict:
        return {
            "name": "Hydrogen-Bromine Flow Battery (HBrFB)",
            "ec_id": "EC040",
            "fidelity": "F1b",
            "description": "Nernst+Arrhenius-R thermal model; 30-cell stack, pump losses, 20-60 degC operating range",
            "inputs": {
                "soc": {"unit": "dimensionless", "range": [0.05, 0.95]},
                "current": {"unit": "A", "range": [-150.0, 150.0],
                            "note": "positive=discharge, negative=charge; high I possible due to fast H2 kinetics"},
                "temperature": {"unit": "K", "range": [293.15, 333.15],
                                "note": "20 to 60 degC"},
            },
            "outputs": {
                "stack_voltage": {"unit": "V"},
                "cell_voltage": {"unit": "V"},
                "power": {"unit": "W", "note": "net: electrical minus pump losses"},
                "heat_generation": {"unit": "W"},
                "pump_loss": {"unit": "W"},
                "internal_resistance_cell": {"unit": "Ohm"},
                "e_nernst": {"unit": "V"},
                "efficiency": {"unit": "dimensionless"},
            },
            "source": "Livshits et al. (2006); Cho et al. (2012); Tucker et al. (2015)",
        }


if __name__ == "__main__":
    model = ComponentModel()
    print(json.dumps(model.get_info(), indent=2))
    result = model.predict({"soc": 0.5, "current": 80.0, "temperature": 313.15})
    print(f"\nAt SOC=0.5, I=80A, T=313.15K (40 degC):")
    for k, v in result.items():
        val = float(v) if np.ndim(v) == 0 else v
        print(f"  {k}: {val:.4f}")
