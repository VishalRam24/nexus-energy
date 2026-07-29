"""EC139 — Salinity Gradient (PRO) — F1a — Standardized Predict Interface"""

import json, numpy as np
from pathlib import Path
from model import SalinityGradientPROF1a


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = SalinityGradientPROF1a(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        inputs:
          C_sw       : seawater concentration [g/L] (default 35)
          C_fw       : freshwater concentration [g/L] (default 0.5)
          Q_feed_m3s : total feed flow rate [m^3/s] (optional)
        returns:
          osmotic_pressure_bar, gibbs_energy_kwh_per_m3,
          net_energy_kwh_per_m3, power_kw, extraction_efficiency
        """
        C_sw = np.asarray(inputs.get("C_sw", self._model.C_sw), dtype=float)
        C_fw = np.asarray(inputs.get("C_fw", self._model.C_fw), dtype=float)
        Q    = inputs.get("Q_feed_m3s", None)

        Pi   = self._model.osmotic_pressure_pa(C_sw, C_fw) / 1e5  # bar
        dG   = self._model.gibbs_energy_kwh_per_m3(C_sw, C_fw)
        w_net = self._model.net_energy_kwh_per_m3(C_sw, C_fw)
        P    = self._model.power_kw(C_sw, C_fw, Q)
        eta  = np.where(dG > 0, w_net / dG, 0.0)

        return {
            "osmotic_pressure_bar":       Pi,
            "gibbs_energy_kwh_per_m3":    dG,
            "net_energy_kwh_per_m3":      w_net,
            "power_kw":                   P,
            "extraction_efficiency":      eta,
        }

    def get_info(self) -> dict:
        return {
            "name":        "Salinity Gradient — Pressure Retarded Osmosis (PRO)",
            "ec_id":       "EC139",
            "fidelity":    "F1a",
            "description": "Π=ν*R*T*ΔC; ΔG=0.5*Π*recovery; w_net=ΔG*eta_mem*eta_turb*eta_px",
            "inputs": {
                "C_sw":       {"unit": "g/L", "range": [25.0, 40.0], "default": 35.0},
                "C_fw":       {"unit": "g/L", "range": [0.1, 2.0],   "default": 0.5},
                "Q_feed_m3s": {"unit": "m^3/s", "range": [0.0, None], "default": "from params"},
            },
            "outputs": {
                "osmotic_pressure_bar":    {"unit": "bar"},
                "gibbs_energy_kwh_per_m3": {"unit": "kWh/m3_freshwater"},
                "net_energy_kwh_per_m3":   {"unit": "kWh/m3_freshwater"},
                "power_kw":                {"unit": "kW"},
                "extraction_efficiency":   {"unit": "-"},
            },
            "source": "Yip & Elimelech (2012) Environ. Sci. Technol.; Straub et al. (2016) Nature Energy",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    print(json.dumps(model.get_info(), indent=2))
    r = model.predict({})
    print(f"\nDefault (C_sw=35 g/L, C_fw=0.5 g/L):")
    print(f"  Osmotic pressure: {float(r['osmotic_pressure_bar']):.1f} bar")
    print(f"  Gibbs energy:     {float(r['gibbs_energy_kwh_per_m3']):.4f} kWh/m³")
    print(f"  Net energy:       {float(r['net_energy_kwh_per_m3']):.4f} kWh/m³")
    print(f"  Power (1 m³/s):   {float(r['power_kw']):.2f} kW")
    print(f"  Extraction eta:   {float(r['extraction_efficiency']):.3f}")
