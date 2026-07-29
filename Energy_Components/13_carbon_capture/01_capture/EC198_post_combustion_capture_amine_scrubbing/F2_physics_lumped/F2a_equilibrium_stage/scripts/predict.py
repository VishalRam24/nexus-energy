"""
EC198 -- Post-Combustion Capture (Amine Scrubbing) -- F2a Equilibrium Stage -- predict interface.
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from model import AmineCapture_F2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH) as f:
        return json.load(f)


class ComponentModel:
    component_id = "EC198"
    component_name = "Post-Combustion Capture (Amine Scrubbing)"
    fidelity = "F2a -- Equilibrium Stage (Absorber + Stripper)"
    version = "1.0.0"

    def __init__(self, params=None):
        self._raw = _load_params()
        if params:
            self._raw["unit"].update(params)
        self._model = AmineCapture_F2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        inputs:
            y_CO2_in       : float  -- CO2 mole fraction in flue gas (default 0.12)
            L_G            : float  -- liquid/gas molar ratio (default 2.5)
            flue_gas_kg_s  : float  -- flue gas mass flow [kg/s] (default 600)
            N_stages       : int    -- absorber equilibrium stages (default 12)
            T_abs_K        : float  -- absorber temperature [K] (default 313.15)
        """
        y_CO2 = inputs.get("y_CO2_in", None)
        L_G = inputs.get("L_G", 2.5)
        fg = inputs.get("flue_gas_kg_s", None)
        N = inputs.get("N_stages", None)
        T = inputs.get("T_abs_K", None)

        return self._model.compute(
            y_CO2_in=y_CO2, L_G=L_G, flue_gas_kg_s=fg,
            N_stages=N, T_abs=T,
        )

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "y_CO2_in": {"unit": "mol/mol", "range": [0.04, 0.15],
                             "note": "CO2 mole fraction in flue gas"},
                "L_G": {"unit": "mol/mol", "range": [1.0, 6.0],
                        "note": "Liquid-to-gas molar ratio"},
                "flue_gas_kg_s": {"unit": "kg/s", "range": [100, 1000],
                                  "note": "Flue gas mass flow rate"},
                "N_stages": {"unit": "-", "range": [4, 30],
                             "note": "Number of absorber equilibrium stages"},
                "T_abs_K": {"unit": "K", "range": [300, 340],
                            "note": "Absorber temperature"},
            },
            "outputs": {
                "capture_rate": "-",
                "y_CO2_out": "mol/mol",
                "rich_loading": "mol_CO2/mol_MEA",
                "Q_reboiler_MW": "MW",
                "SRD_GJ_per_tCO2": "GJ/tCO2",
                "CO2_captured_kg_s": "kg/s",
                "CO2_captured_t_per_year": "t/year",
                "electricity_MW": "MW",
                "total_energy_MW": "MW",
                "total_specific_energy_GJ_per_tCO2": "GJ/tCO2",
            },
            "source": self._raw.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    r = m.predict({"y_CO2_in": 0.12, "L_G": 2.5})
    print(f"Capture rate:  {r['capture_rate']*100:.1f}%")
    print(f"SRD:           {r['SRD_GJ_per_tCO2']:.2f} GJ/tCO2")
    print(f"Q_reboiler:    {r['Q_reboiler_MW']:.1f} MW")
    print(f"CO2 captured:  {r['CO2_captured_t_per_year']/1e6:.2f} MtCO2/year")
