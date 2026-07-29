"""EC152 -- Flash Steam Geothermal Plant -- F1b Part-Load Ambient -- Standardized Predict Interface"""
import json
import numpy as np
from pathlib import Path
from model import FlashSteamGeothermalF1b


class ComponentModel:
    """Standardized interface for EC152 Flash Steam Geothermal -- F1b."""

    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = FlashSteamGeothermalF1b(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        Args:
            inputs: {
                "T_brine_degC":       float [150-350]  -- wellhead brine temperature
                "m_dot_brine_kg_s":   float [10-500]   -- brine mass flow rate
                "T_reject_degC":      float [5-50]     -- cooling rejection temperature
                "PLR":                float [0.3-1.0]  (default 1.0)
                "years_operation":    float [0-50]     (default 0)
                "TDS_g_L":            float [0.5-100]  total dissolved solids (default: base 10)
            }

        Returns:
            dict with: power_output_kw, efficiency, resource_factor, condenser_factor,
                       scaling_factor, flash_config, steam_quality
        """
        return self._model.predict(
            T_brine_degC=float(inputs.get("T_brine_degC", 240.0)),
            m_dot_brine_kg_s=float(inputs.get("m_dot_brine_kg_s", 100.0)),
            T_reject_degC=float(inputs.get("T_reject_degC", 30.0)),
            PLR=float(inputs.get("PLR", 1.0)),
            years_operation=float(inputs.get("years_operation", 0.0)),
            TDS_g_L=inputs.get("TDS_g_L", None),
        )

    def get_info(self) -> dict:
        m = self._model
        return {
            "name": "Flash Steam Geothermal Plant",
            "ec_id": "EC152",
            "fidelity": "F1b",
            "model": "Part-Load Turbine + Ambient Derating + Brine Scaling + Resource Decline + Double Flash",
            "description": (
                f"Flash steam plant with part-load curve, condenser ambient derating "
                f"({m.cooling_mode} cooling), brine chemistry scaling, "
                f"{m.decline_rate*100:.1f}%/yr resource decline, and double-flash bonus "
                f"(+{m.double_flash_bonus*100:.0f}%) above {m.T_double_flash_min:.0f} degC. "
                f"Rated: {m.P_rated:.0f} kW."
            ),
            "inputs": {
                "T_brine_degC":     {"unit": "degC", "range": [150.0, 350.0]},
                "m_dot_brine_kg_s": {"unit": "kg/s", "range": [10.0, 500.0]},
                "T_reject_degC":    {"unit": "degC", "range": [5.0, 50.0]},
                "PLR":              {"unit": "dimensionless", "range": [0.3, 1.0], "default": 1.0},
                "years_operation":  {"unit": "years", "range": [0.0, 50.0], "default": 0.0},
                "TDS_g_L":          {"unit": "g/L", "range": [0.5, 100.0], "default": 10.0},
            },
            "outputs": {
                "power_output_kw":  {"unit": "kW"},
                "efficiency":       {"unit": "dimensionless"},
                "resource_factor":  {"unit": "dimensionless"},
                "condenser_factor": {"unit": "dimensionless"},
                "scaling_factor":   {"unit": "dimensionless"},
                "flash_config":     {"unit": "dimensionless"},
                "steam_quality":    {"unit": "dimensionless"},
            },
            "source": "DiPippo (2015); Zarrouk & Moon (2014); Vaca-Mier et al. (2003)",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    info = model.get_info()
    print(f"{info['ec_id']} -- {info['name']} -- {info['fidelity']}")

    # Single flash, design conditions
    r = model.predict({
        "T_brine_degC": 200.0, "m_dot_brine_kg_s": 100.0,
        "T_reject_degC": 30.0, "PLR": 1.0,
    })
    print("\nSingle flash, design:")
    for k, v in r.items():
        print(f"  {k}: {v:.4f}")

    # Double flash, high T
    r2 = model.predict({
        "T_brine_degC": 240.0, "m_dot_brine_kg_s": 100.0,
        "T_reject_degC": 30.0, "PLR": 1.0,
    })
    print("\nDouble flash eligible (240 degC):")
    for k, v in r2.items():
        print(f"  {k}: {v:.4f}")

    # Degraded: high TDS, hot ambient, 20 years
    r3 = model.predict({
        "T_brine_degC": 240.0, "m_dot_brine_kg_s": 100.0,
        "T_reject_degC": 40.0, "PLR": 0.7,
        "years_operation": 20.0, "TDS_g_L": 25.0,
    })
    print("\nDegraded (20yr, TDS=25, T_rej=40, PLR=0.7):")
    for k, v in r3.items():
        print(f"  {k}: {v:.4f}")
