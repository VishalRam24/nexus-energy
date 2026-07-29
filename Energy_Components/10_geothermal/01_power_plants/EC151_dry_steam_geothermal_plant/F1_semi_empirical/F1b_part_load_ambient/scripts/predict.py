"""EC151 -- Dry Steam Geothermal Plant -- F1b Part-Load Ambient -- Standardized Predict Interface"""
import json
import numpy as np
from pathlib import Path
from model import DrySteamGeothermalF1b


class ComponentModel:
    """Standardized interface for EC151 Dry Steam Geothermal -- F1b part-load model."""

    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = DrySteamGeothermalF1b(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        Args:
            inputs: {
                "T_geo_degC":          float [150-280]  -- steam wellhead temperature
                "m_dot_steam_kg_s":    float [5-200]    -- steam mass flow rate
                "T_reject_degC":       float [5-50]     -- cooling rejection temperature
                "PLR":                 float [0.3-1.0]  (default 1.0)
                "years_operation":     float [0-50]     (default 0)
                "ncg_content_pct":     float [0-10]     wt% NCG (default: base value 1.0)
            }

        Returns:
            dict with: power_output_kw, efficiency, resource_factor,
                       condenser_factor, ncg_factor, plr_factor
        """
        return self._model.predict(
            T_geo_degC=float(inputs.get("T_geo_degC", 200.0)),
            m_dot_steam_kg_s=float(inputs.get("m_dot_steam_kg_s", 50.0)),
            T_reject_degC=float(inputs.get("T_reject_degC", 30.0)),
            PLR=float(inputs.get("PLR", 1.0)),
            years_operation=float(inputs.get("years_operation", 0.0)),
            ncg_content_pct=inputs.get("ncg_content_pct", None),
        )

    def get_info(self) -> dict:
        m = self._model
        return {
            "name": "Dry Steam Geothermal Plant",
            "ec_id": "EC151",
            "fidelity": "F1b",
            "model": "Part-Load Turbine + Ambient Derating + NCG Penalty + Resource Decline",
            "description": (
                f"Dry steam plant with part-load efficiency curve, condenser ambient "
                f"derating ({m.cooling_mode} cooling), NCG back-pressure penalty, and "
                f"{m.decline_rate*100:.1f}%/yr steam resource decline. "
                f"Rated: {m.P_rated:.0f} kW at T_geo={m.T_geo_design:.0f} degC."
            ),
            "inputs": {
                "T_geo_degC":       {"unit": "degC",         "range": [150.0, 280.0]},
                "m_dot_steam_kg_s": {"unit": "kg/s",         "range": [5.0, 200.0]},
                "T_reject_degC":    {"unit": "degC",         "range": [5.0, 50.0]},
                "PLR":              {"unit": "dimensionless","range": [0.3, 1.0], "default": 1.0},
                "years_operation":  {"unit": "years",        "range": [0.0, 50.0], "default": 0.0},
                "ncg_content_pct":  {"unit": "wt%",          "range": [0.0, 10.0], "default": 1.0},
            },
            "outputs": {
                "power_output_kw":  {"unit": "kW"},
                "efficiency":       {"unit": "dimensionless"},
                "resource_factor":  {"unit": "dimensionless"},
                "condenser_factor": {"unit": "dimensionless"},
                "ncg_factor":       {"unit": "dimensionless"},
                "plr_factor":       {"unit": "dimensionless"},
            },
            "source": "DiPippo (2015); Zarrouk & Moon (2014); Sutton (1976)",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    info = model.get_info()
    print(f"{info['ec_id']} -- {info['name']} -- {info['fidelity']}")

    # Design conditions
    r = model.predict({
        "T_geo_degC": 200.0, "m_dot_steam_kg_s": 50.0,
        "T_reject_degC": 30.0, "PLR": 1.0, "years_operation": 0.0,
    })
    print("\nDesign conditions:")
    for k, v in r.items():
        print(f"  {k}: {v:.4f}")

    # After 20 years, high NCG
    r2 = model.predict({
        "T_geo_degC": 200.0, "m_dot_steam_kg_s": 50.0,
        "T_reject_degC": 35.0, "PLR": 0.8,
        "years_operation": 20.0, "ncg_content_pct": 3.0,
    })
    print("\n20 years, T_rej=35, PLR=0.8, NCG=3%:")
    for k, v in r2.items():
        print(f"  {k}: {v:.4f}")
