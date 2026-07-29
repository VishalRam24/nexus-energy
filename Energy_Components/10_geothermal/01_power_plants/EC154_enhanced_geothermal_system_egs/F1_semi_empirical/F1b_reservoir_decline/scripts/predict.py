"""EC154 -- EGS -- F1b Reservoir Decline -- Standardized Predict Interface"""
import json
import numpy as np
from pathlib import Path
from model import EGSF1b


class ComponentModel:
    """Standardized interface for EC154 EGS -- F1b reservoir decline model."""

    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = EGSF1b(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        Args:
            inputs: {
                "T_geo_degC":       float [150-350]  -- rock/fluid temperature
                "m_dot_kg_s":       float [5-200]    -- design circulation flow rate
                "T_reject_degC":    float [-10 to 50] -- ambient rejection temperature
                "PLR":              float [0.3-1.0]  (default 1.0)
                "years_operation":  float [0-50]     (default 0)
                "delta_P_MPa":      float [1-20]     reservoir pressure diff (default: design 5 MPa)
            }

        Returns:
            dict with: power_output_kw, gross_efficiency, net_efficiency,
                       resource_factor, permeability_ratio, pump_parasitic_frac,
                       effective_flow_kg_s, T_out_degC
        """
        return self._model.predict(
            T_geo_degC=float(inputs.get("T_geo_degC", 200.0)),
            m_dot_kg_s=float(inputs.get("m_dot_kg_s", 50.0)),
            T_reject_degC=float(inputs.get("T_reject_degC", 25.0)),
            PLR=float(inputs.get("PLR", 1.0)),
            years_operation=float(inputs.get("years_operation", 0.0)),
            delta_P_MPa=inputs.get("delta_P_MPa", None),
        )

    def get_info(self) -> dict:
        m = self._model
        return {
            "name": "Enhanced Geothermal System (EGS)",
            "ec_id": "EC154",
            "fidelity": "F1b",
            "model": "Thermal Breakthrough + Permeability Decline + ORC Part-Load + Ambient Derating",
            "description": (
                f"EGS with reservoir thermal breakthrough (tau={m.tau_thermal:.1f} yr), "
                f"stress-induced fracture permeability decline, ORC part-load, and "
                f"air-cooled condenser ambient derating. Rated: {m.P_rated:.0f} kW."
            ),
            "inputs": {
                "T_geo_degC":      {"unit": "degC", "range": [150.0, 350.0]},
                "m_dot_kg_s":      {"unit": "kg/s", "range": [5.0, 200.0]},
                "T_reject_degC":   {"unit": "degC", "range": [-10.0, 50.0]},
                "PLR":             {"unit": "dimensionless", "range": [0.3, 1.0], "default": 1.0},
                "years_operation": {"unit": "years", "range": [0.0, 50.0], "default": 0.0},
                "delta_P_MPa":     {"unit": "MPa", "range": [1.0, 20.0], "default": 5.0},
            },
            "outputs": {
                "power_output_kw":     {"unit": "kW"},
                "gross_efficiency":    {"unit": "dimensionless"},
                "net_efficiency":      {"unit": "dimensionless"},
                "resource_factor":     {"unit": "dimensionless"},
                "permeability_ratio":  {"unit": "dimensionless"},
                "pump_parasitic_frac": {"unit": "dimensionless"},
                "effective_flow_kg_s": {"unit": "kg/s"},
                "T_out_degC":          {"unit": "degC"},
            },
            "source": "Tester et al. (2006); DiPippo (2015) Ch.16; Sanyal & Butler (2005)",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    info = model.get_info()
    print(f"{info['ec_id']} -- {info['name']} -- {info['fidelity']}")

    # Fresh EGS
    r = model.predict({
        "T_geo_degC": 200.0, "m_dot_kg_s": 50.0,
        "T_reject_degC": 25.0, "PLR": 1.0, "years_operation": 0.0,
    })
    print("\nFresh EGS (year 0):")
    for k, v in r.items():
        print(f"  {k}: {v:.4f}")

    # After 15 years (thermal breakthrough regime)
    r2 = model.predict({
        "T_geo_degC": 200.0, "m_dot_kg_s": 50.0,
        "T_reject_degC": 25.0, "PLR": 0.8, "years_operation": 15.0,
    })
    print("\n15 years (thermal breakthrough + perm decline):")
    for k, v in r2.items():
        print(f"  {k}: {v:.4f}")
