"""EC146 -- Torrefaction Reactor -- F1a Yield Model -- Standardized Predict Interface"""
import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from model import TorrefactionReactorF1a


class ComponentModel:
    component_id = "EC146"

    def __init__(self, params_path=None):
        if params_path is None:
            params_path = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = TorrefactionReactorF1a(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        inputs:
            feedstock_dry_kg_per_h : float  Dry biomass feed [kg/h]
        returns:
            torrefied_solid_kg_per_h, volatile_loss_kg_per_h, energy_in_MW,
            energy_out_MW, energy_yield, LHV_torrefied_MJ_kg, energy_density_factor
        """
        feed = float(inputs.get("feedstock_dry_kg_per_h", 1000.0))
        return self._model.predict(feed)

    def get_info(self) -> dict:
        m = self._model
        return {
            "name":        "Torrefaction Reactor",
            "ec_id":       "EC146",
            "fidelity":    "F1a",
            "model":       "Yield Model (torrefied = feedstock * solid_yield; energy_density *= 1.3x)",
            "description": (
                f"Torrefaction at {m.T_operating:.0f} degC. "
                f"solid_yield={m.solid_yield:.0%}, energy_yield={m.energy_yield:.0%}, "
                f"energy_density_factor={m.energy_density_factor:.1f}x."
            ),
            "inputs":  {"feedstock_dry_kg_per_h": {"unit": "kg/h", "range": [0.0, 1e5]}},
            "outputs": {
                "torrefied_solid_kg_per_h": {"unit": "kg/h"},
                "volatile_loss_kg_per_h":   {"unit": "kg/h"},
                "energy_in_MW":             {"unit": "MW"},
                "energy_out_MW":            {"unit": "MW"},
                "energy_yield":             {"unit": "dimensionless"},
                "LHV_torrefied_MJ_kg":      {"unit": "MJ/kg"},
                "energy_density_factor":    {"unit": "dimensionless"},
            },
            "source": "Bergman et al. (2005) ECN-C-05-073; van der Stelt et al. (2011) BB 35:3748",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    info = model.get_info()
    print(f"{info['ec_id']} -- {info['name']} -- {info['fidelity']}")
    r = model.predict({"feedstock_dry_kg_per_h": 1000.0})
    for k, v in r.items():
        print(f"  {k}: {v:.3f}")
