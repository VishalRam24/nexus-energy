"""EC155 -- Geothermal District Heating -- F1b Resource Decline & Part-Load -- Standardized Interface"""
import json
import numpy as np
from pathlib import Path
from model import GeothermalDistrictHeatingF1b


class ComponentModel:
    """Standardized interface for EC155 Geothermal District Heating -- F1b."""

    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = GeothermalDistrictHeatingF1b(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        Args:
            inputs: {
                "T_source_degC":     float [50-150]  -- geothermal supply temperature
                "m_dot_geo_kg_s":    float [5-200]   -- geothermal flow rate
                "T_return_degC":     float [20-60]   -- reinjection return temperature
                "T_net_supply_degC": float [40-90]   -- district network supply temperature
                "PLR":               float [0.1-1.0] (default 1.0)
                "years_operation":   float [0-50]    (default 0)
            }

        Returns:
            dict with: heat_delivered_kw, heat_extracted_kw, pump_power_kw,
                       system_cop, T_source_effective, resource_factor,
                       distribution_loss_frac
        """
        return self._model.predict(
            T_source_degC=float(inputs.get("T_source_degC", 80.0)),
            m_dot_geo_kg_s=float(inputs.get("m_dot_geo_kg_s", 50.0)),
            T_return_degC=float(inputs.get("T_return_degC", 40.0)),
            T_net_supply_degC=float(inputs.get("T_net_supply_degC", 70.0)),
            PLR=float(inputs.get("PLR", 1.0)),
            years_operation=float(inputs.get("years_operation", 0.0)),
        )

    def get_info(self) -> dict:
        m = self._model
        return {
            "name": "Geothermal District Heating",
            "ec_id": "EC155",
            "fidelity": "F1b",
            "model": "Aquifer Decline + Seasonal Part-Load + Network T + Distribution Loss",
            "description": (
                f"Direct-use geothermal district heating with {m.decline_rate:.1f} degC/yr "
                f"aquifer temperature decline, seasonal part-load efficiency, "
                f"network supply temperature sensitivity, and temperature-dependent "
                f"pipe distribution losses."
            ),
            "inputs": {
                "T_source_degC":     {"unit": "degC", "range": [50.0, 150.0]},
                "m_dot_geo_kg_s":    {"unit": "kg/s", "range": [5.0, 200.0]},
                "T_return_degC":     {"unit": "degC", "range": [20.0, 60.0]},
                "T_net_supply_degC": {"unit": "degC", "range": [40.0, 90.0]},
                "PLR":               {"unit": "dimensionless", "range": [0.1, 1.0], "default": 1.0},
                "years_operation":   {"unit": "years", "range": [0.0, 50.0], "default": 0.0},
            },
            "outputs": {
                "heat_delivered_kw":      {"unit": "kW"},
                "heat_extracted_kw":      {"unit": "kW"},
                "pump_power_kw":          {"unit": "kW"},
                "system_cop":             {"unit": "dimensionless"},
                "T_source_effective":     {"unit": "degC"},
                "resource_factor":        {"unit": "dimensionless"},
                "distribution_loss_frac": {"unit": "dimensionless"},
            },
            "source": "Lund & Toth (2021); Rybach & Mongillo (2006)",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    info = model.get_info()
    print(f"{info['ec_id']} -- {info['name']} -- {info['fidelity']}")

    # Design conditions
    r = model.predict({
        "T_source_degC": 80.0, "m_dot_geo_kg_s": 50.0,
        "T_return_degC": 40.0, "T_net_supply_degC": 70.0,
        "PLR": 1.0, "years_operation": 0.0,
    })
    print("\nDesign conditions:")
    for k, v in r.items():
        print(f"  {k}: {v:.4f}")

    # After 20 years, summer part-load
    r2 = model.predict({
        "T_source_degC": 80.0, "m_dot_geo_kg_s": 50.0,
        "T_return_degC": 40.0, "T_net_supply_degC": 60.0,
        "PLR": 0.3, "years_operation": 20.0,
    })
    print("\n20 years, summer (PLR=0.3, T_net=60):")
    for k, v in r2.items():
        print(f"  {k}: {v:.4f}")
