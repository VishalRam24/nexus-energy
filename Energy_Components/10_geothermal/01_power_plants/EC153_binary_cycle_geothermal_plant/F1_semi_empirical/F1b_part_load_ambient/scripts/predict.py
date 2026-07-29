"""EC153 -- Binary Cycle Geothermal -- F1b Part-Load Ambient -- Standardized Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import BinaryGeothermalF1b


class ComponentModel:
    """Standardized interface for EC153 Binary Geothermal -- F1b part-load model."""

    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = BinaryGeothermalF1b(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        Args:
            inputs: {
                "T_brine_degC":      float [80-200]
                "brine_flow_kg_s":   float [10-200]
                "T_ambient_degC":    float [-20 to 50]
                "PLR":               float [0.3-1.0] (default 1.0)
                "years_operation":   float [0-50] (default 0)
            }

        Returns:
            dict with: power_output_kw, efficiency, resource_factor, condenser_factor
        """
        return self._model.predict(
            T_brine_degC=float(inputs.get("T_brine_degC", 150.0)),
            brine_flow_kg_s=float(inputs.get("brine_flow_kg_s", 80.0)),
            T_ambient_degC=float(inputs.get("T_ambient_degC", 25.0)),
            PLR=float(inputs.get("PLR", 1.0)),
            years_operation=float(inputs.get("years_operation", 0.0)),
        )

    def get_info(self) -> dict:
        m = self._model
        return {
            "name": "Binary Cycle Geothermal Plant",
            "ec_id": "EC153",
            "fidelity": "F1b",
            "model": "Part-Load ORC with Ambient Derating & Resource Decline",
            "description": (
                f"Binary ORC with part-load efficiency curve, air-cooled condenser "
                f"ambient derating, and {m.decline_rate*100:.1f}%/yr resource decline. "
                f"Rated: {m.P_rated:.0f} kW at T_brine={m.T_brine_design:.0f} degC, "
                f"T_cond={m.T_cond_design:.0f} degC, eta={m.eta_design:.2f}."
            ),
            "inputs": {
                "T_brine_degC":    {"unit": "degC", "range": [80.0, 200.0]},
                "brine_flow_kg_s": {"unit": "kg/s", "range": [10.0, 200.0]},
                "T_ambient_degC":  {"unit": "degC", "range": [-20.0, 50.0]},
                "PLR":             {"unit": "dimensionless", "range": [0.3, 1.0], "default": 1.0},
                "years_operation": {"unit": "years", "range": [0.0, 50.0], "default": 0.0},
            },
            "outputs": {
                "power_output_kw":  {"unit": "kW"},
                "efficiency":       {"unit": "dimensionless"},
                "resource_factor":  {"unit": "dimensionless"},
                "condenser_factor": {"unit": "dimensionless"},
            },
            "source": "DiPippo (2015); Lukawski et al. (2014)",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    info = model.get_info()
    print(f"{info['ec_id']} -- {info['name']} -- {info['fidelity']}")

    # Design conditions
    r = model.predict({
        "T_brine_degC": 150.0, "brine_flow_kg_s": 80.0,
        "T_ambient_degC": 30.0, "PLR": 1.0, "years_operation": 0.0,
    })
    print(f"\nDesign conditions:")
    for k, v in r.items():
        print(f"  {k}: {v:.4f}")

    # After 20 years, hot day
    r2 = model.predict({
        "T_brine_degC": 150.0, "brine_flow_kg_s": 80.0,
        "T_ambient_degC": 40.0, "PLR": 0.8, "years_operation": 20.0,
    })
    print(f"\n20 years, T_amb=40, PLR=0.8:")
    for k, v in r2.items():
        print(f"  {k}: {v:.4f}")
