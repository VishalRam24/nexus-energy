"""EC142 -- Biogas Upgrading to Biomethane -- F1a Yield Model -- Standardized Predict Interface"""
import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from model import BiogasUpgradingF1a


class ComponentModel:
    component_id = "EC142"

    def __init__(self, params_path=None):
        if params_path is None:
            params_path = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = BiogasUpgradingF1a(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        inputs:
            biogas_flow_Nm3_per_h : float  Raw biogas flow [Nm3/h]
            CH4_fraction_in       : float  CH4 content of raw biogas [-] (default 0.60)
        returns:
            biomethane_Nm3_per_h, CH4_recovered_Nm3_per_h, electricity_kW,
            energy_output_kW, CH4_recovery, biomethane_purity
        """
        Q  = float(inputs.get("biogas_flow_Nm3_per_h", 100.0))
        x  = float(inputs.get("CH4_fraction_in", 0.60))
        return self._model.predict(Q, x)

    def get_info(self) -> dict:
        m = self._model
        return {
            "name":        "Biogas Upgrading to Biomethane",
            "ec_id":       "EC142",
            "fidelity":    "F1a",
            "model":       "Yield Model (biomethane = biogas * CH4_fraction * CH4_recovery)",
            "description": (
                f"Biogas upgrading. CH4_recovery={m.CH4_recovery:.0%}, "
                f"purity={m.biomethane_purity:.0%}, SEC={m.SEC_kWh_per_Nm3:.2f} kWh/Nm3."
            ),
            "inputs": {
                "biogas_flow_Nm3_per_h": {"unit": "Nm3/h",       "range": [0.0, 1e5]},
                "CH4_fraction_in":       {"unit": "dimensionless","range": [0.4, 0.75]},
            },
            "outputs": {
                "biomethane_Nm3_per_h":    {"unit": "Nm3/h"},
                "CH4_recovered_Nm3_per_h": {"unit": "Nm3/h"},
                "electricity_kW":          {"unit": "kW"},
                "energy_output_kW":        {"unit": "kW"},
                "CH4_recovery":            {"unit": "dimensionless"},
                "biomethane_purity":       {"unit": "dimensionless"},
            },
            "source": "IEA Bioenergy Task 37 (2017); Bauer et al. (2013) BRT 122:145",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    info = model.get_info()
    print(f"{info['ec_id']} -- {info['name']} -- {info['fidelity']}")
    r = model.predict({"biogas_flow_Nm3_per_h": 100.0, "CH4_fraction_in": 0.60})
    for k, v in r.items():
        print(f"  {k}: {v:.3f}")
