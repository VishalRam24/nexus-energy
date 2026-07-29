"""EC150 -- Fischer-Tropsch Synthesis (BTL) -- F1a Yield Model -- Standardized Predict Interface"""
import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from model import FischerTropschF1a


class ComponentModel:
    component_id = "EC150"

    def __init__(self, params_path=None):
        if params_path is None:
            params_path = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = FischerTropschF1a(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        inputs:
            syngas_flow_Nm3_per_h : float  Total syngas flow [Nm3/h]
            CO_fraction_in        : float  CO mole fraction in syngas [-] (default 0.40)
        returns:
            CO_reacted_Nm3_per_h, FT_liquid_kg_per_h, diesel_kg_per_h,
            naphtha_kg_per_h, wax_kg_per_h, light_gas_kg_per_h,
            energy_output_MW, CO_conversion
        """
        Q  = float(inputs.get("syngas_flow_Nm3_per_h", 1000.0))
        xCO = float(inputs.get("CO_fraction_in", 0.40))
        return self._model.predict(Q, xCO)

    def get_info(self) -> dict:
        m = self._model
        return {
            "name":        "Fischer-Tropsch Synthesis (BTL)",
            "ec_id":       "EC150",
            "fidelity":    "F1a",
            "model":       "Yield Model (FT liquid = CO_converted * ASF_liquid_fraction * yield)",
            "description": (
                f"Low-temp FT at {m.T_operating:.0f} degC / {m.P_operating:.0f} bar. "
                f"CO_conversion={m.CO_conversion:.0%}, alpha_ASF={m.alpha_ASF:.2f}. "
                f"LHV_FT_liquid={m.LHV_FT:.0f} MJ/kg."
            ),
            "inputs": {
                "syngas_flow_Nm3_per_h": {"unit": "Nm3/h",       "range": [0.0, 1e6]},
                "CO_fraction_in":        {"unit": "dimensionless","range": [0.2, 0.6]},
            },
            "outputs": {
                "CO_reacted_Nm3_per_h": {"unit": "Nm3/h"},
                "FT_liquid_kg_per_h":   {"unit": "kg/h"},
                "diesel_kg_per_h":      {"unit": "kg/h"},
                "naphtha_kg_per_h":     {"unit": "kg/h"},
                "wax_kg_per_h":         {"unit": "kg/h"},
                "light_gas_kg_per_h":   {"unit": "kg/h"},
                "energy_output_MW":     {"unit": "MW"},
                "CO_conversion":        {"unit": "dimensionless"},
            },
            "source": "Dry (2002) Catal. Today 71:227; Steynberg & Dry (2004); Spath & Dayton NREL/TP-510-34929",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    info = model.get_info()
    print(f"{info['ec_id']} -- {info['name']} -- {info['fidelity']}")
    r = model.predict({"syngas_flow_Nm3_per_h": 1000.0, "CO_fraction_in": 0.40})
    for k, v in r.items():
        print(f"  {k}: {v:.3f}")
