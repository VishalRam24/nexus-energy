"""EC104 — Gas Engine CHP — F1a Part-Load Model — Standardized Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import GasEngineCHPF1a


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = GasEngineCHPF1a(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        inputs:
            part_load_ratio : float or array — PLR [0.5–1.0]
        returns:
            electrical_power_kw : kW_e
            thermal_power_kw    : kW_th
            fuel_input_kw       : kW (LHV basis)
            eta_electrical      : dimensionless
            eta_thermal         : dimensionless
            eta_total           : dimensionless
        """
        plr = np.asarray(inputs["part_load_ratio"], dtype=float)
        return self._model.compute(plr)

    def get_info(self) -> dict:
        return {
            "name": "Gas Engine CHP",
            "ec_id": "EC104",
            "fidelity": "F1a",
            "description": "Part-load efficiency model: eta_el(PLR), eta_th(PLR), fuel and power flows",
            "inputs": {
                "part_load_ratio": {"unit": "-", "range": [0.5, 1.0]},
            },
            "outputs": {
                "electrical_power_kw": {"unit": "kW_e"},
                "thermal_power_kw":    {"unit": "kW_th"},
                "fuel_input_kw":       {"unit": "kW (LHV)"},
                "eta_electrical":      {"unit": "-"},
                "eta_thermal":         {"unit": "-"},
                "eta_total":           {"unit": "-"},
            },
            "source": "US EPA CHP Catalog (2017); ASUE BHKW-Kenndaten (2011)",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    r = model.predict({"part_load_ratio": 1.0})
    print(f"Full load: P_el={float(r['electrical_power_kw']):.0f}kW, "
          f"Q_th={float(r['thermal_power_kw']):.0f}kW, "
          f"fuel={float(r['fuel_input_kw']):.0f}kW, "
          f"eta_el={float(r['eta_electrical']):.3f}, "
          f"eta_th={float(r['eta_thermal']):.3f}, "
          f"eta_tot={float(r['eta_total']):.3f}")
