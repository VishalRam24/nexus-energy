"""EC139 — Salinity Gradient (Blue Energy, PRO) — F0a — standardized predict interface."""
import json, numpy as np
from pathlib import Path
from model import SalinityGradientF0a


class ComponentModel:
    component_id = "EC139"
    component_name = "Salinity Gradient (Blue Energy, PRO)"
    fidelity = "F0a — empirical lookup"
    version = "1.0.0"

    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = SalinityGradientF0a(self.params)

    def predict(self, inputs: dict) -> dict:
        Csw = np.asarray(inputs["C_seawater_g_per_L"], dtype=float)
        return {
            "specific_energy_kWh_per_m3": self._model.specific_energy_kwh_m3(Csw),
            "net_power_kw": self._model.net_power_kw(Csw),
        }

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "C_seawater_g_per_L": {"unit": "g/L", "range": [25.0, 40.0],
                                       "note": "Seawater concentration; freshwater fixed at 0.5 g/L"},
            },
            "outputs": {
                "specific_energy_kWh_per_m3": {"unit": "kWh/m3"},
                "net_power_kw": {"unit": "kW"},
            },
            "source": self.params["source"],
        }


if __name__ == "__main__":
    m = ComponentModel()
    r = m.predict({"C_seawater_g_per_L": 35.0})
    print(f"At Csw=35 g/L: SE={float(r['specific_energy_kWh_per_m3']):.4f} kWh/m3, P={float(r['net_power_kw']):.1f} kW")
