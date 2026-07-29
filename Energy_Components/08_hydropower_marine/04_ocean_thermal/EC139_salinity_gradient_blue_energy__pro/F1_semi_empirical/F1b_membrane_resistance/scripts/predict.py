"""EC139 -- Salinity Gradient (PRO) -- F1b Membrane Resistance -- Standard Predict Interface"""

import json, numpy as np
from pathlib import Path
from model import SalinityGradientPROF1b


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = SalinityGradientPROF1b(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        inputs:
          C_sw      : seawater concentration [g/L]  (default 35)
          C_fw      : freshwater concentration [g/L] (default 0.5)
          dP_bar    : applied hydraulic pressure [bar] (default 12)
          T_degC    : operating temperature [degC] (default 25)
        returns:
          J_w_m_s, dPi_eff_bar, power_density_W_m2,
          net_energy_kwh_per_m3, power_kw, cp_factor_ICP, cp_factor_ECP
        """
        C_sw   = inputs.get("C_sw",   None)
        C_fw   = inputs.get("C_fw",   None)
        dP_bar = inputs.get("dP_bar", None)
        T_degC = inputs.get("T_degC", None)

        result = self._model.net_energy_kwh_per_m3_fw(
            C_sw=C_sw, C_fw=C_fw, dP_bar=dP_bar, T_degC=T_degC
        )
        return {k: (float(v) if np.ndim(v) == 0 else v) for k, v in result.items()}

    def get_info(self) -> dict:
        return {
            "name":        "Salinity Gradient -- Pressure Retarded Osmosis (PRO)",
            "ec_id":       "EC139",
            "fidelity":    "F1b",
            "description": (
                "A-B membrane transport + ICP/ECP concentration polarization "
                "+ temperature-dependent diffusivity. Energy basis: per m3 freshwater "
                "(Yip & Elimelech 2012 Phase 7)."
            ),
            "inputs": {
                "C_sw":   {"unit": "g/L",  "range": [25.0, 40.0], "default": 35.0},
                "C_fw":   {"unit": "g/L",  "range": [0.1,  2.0],  "default": 0.5},
                "dP_bar": {"unit": "bar",  "range": [0.0,  30.0], "default": 12.0},
                "T_degC": {"unit": "degC", "range": [0.0,  40.0], "default": 25.0},
            },
            "outputs": {
                "J_w_m_s":               {"unit": "m/s",            "description": "Water flux through membrane"},
                "dPi_eff_bar":           {"unit": "bar",            "description": "Effective osmotic pressure after CP"},
                "power_density_W_m2":    {"unit": "W/m2",           "description": "Gross power density"},
                "net_energy_kwh_per_m3": {"unit": "kWh/m3_fw",      "description": "Net energy per m3 freshwater (Phase 7 basis)"},
                "power_kw":              {"unit": "kW",             "description": "Net electrical power output"},
                "cp_factor_ICP":         {"unit": "-",              "description": "ICP concentration factor (C_fs/C_fw)"},
                "cp_factor_ECP":         {"unit": "-",              "description": "ECP concentration factor (C_ds/C_sw)"},
            },
            "adds_over_F1a": [
                "Membrane A-B resistance (A_w, B permeability)",
                "Internal concentration polarization (ICP) via structural parameter S",
                "External concentration polarization (ECP) via mass transfer k_d",
                "Temperature-dependent diffusivity (Stokes-Einstein)",
                "Pump parasitic subtraction with pressure exchanger recovery",
            ],
            "source": "Yip & Elimelech (2012); Achilli & Childress (2010); Straub et al. (2016)",
        }


if __name__ == "__main__":
    model = ComponentModel()
    print(json.dumps(model.get_info(), indent=2))
    r = model.predict({})
    print(f"\nDefault (C_sw=35 g/L, C_fw=0.5 g/L, T=25 degC, dP=12 bar):")
    print(f"  Water flux J_w:   {r['J_w_m_s']:.2e} m/s")
    print(f"  Effective dPi:    {r['dPi_eff_bar']:.2f} bar")
    print(f"  Power density:    {r['power_density_W_m2']:.2f} W/m2")
    print(f"  Net energy:       {r['net_energy_kwh_per_m3']:.4f} kWh/m3_fw")
    print(f"  Net power:        {r['power_kw']:.2f} kW")
    print(f"  ICP factor:       {r['cp_factor_ICP']:.3f}")
    print(f"  ECP factor:       {r['cp_factor_ECP']:.3f}")
