"""EC080 -- PCM Storage -- F1b Enthalpy Method -- Standardized Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import PCMStorageF1b


class ComponentModel:
    """Standardized interface for EC080 PCM Storage -- F1b enthalpy method model."""

    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = PCMStorageF1b(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        Args:
            inputs: {
                "T_htf_in_degC":   float, HTF inlet temperature [degC]
                "flow_rate_kg_s":  float, HTF mass flow rate [kg/s]
                "mode":            str, 'charge' | 'discharge' | 'idle'
                "duration_s":      float, step duration [s] (default 3600)
                "T_ambient_degC":  float, ambient temp [degC] (default 25)
                "T_pcm_init":      float, initial PCM temp [degC] (optional)
            }

        Returns:
            dict with: T_pcm_degC, phase_fraction, energy_stored_kwh,
                       thermal_power_kw, T_outlet_degC
        """
        return self._model.predict(
            T_htf_in_degC=float(inputs.get("T_htf_in_degC", 70.0)),
            flow_rate_kg_s=float(inputs.get("flow_rate_kg_s", 0.0)),
            mode=str(inputs.get("mode", "idle")),
            duration_s=float(inputs.get("duration_s", 3600.0)),
            T_ambient_degC=float(inputs.get("T_ambient_degC", 25.0)),
            T_pcm_init=inputs.get("T_pcm_init", None),
        )

    def get_info(self) -> dict:
        m = self._model
        return {
            "name": "Phase-Change Material (PCM) Thermal Energy Storage",
            "ec_id": "EC080",
            "fidelity": "F1b",
            "model": "Enthalpy Method with Mushy Zone",
            "description": (
                f"Enthalpy-method PCM model. Paraffin RT58: T_pc={m.T_pc} degC, "
                f"L={m.L/1000:.0f} kJ/kg, cp_s={m.cp_s/1000:.1f} kJ/(kg*K), "
                f"cp_l={m.cp_l/1000:.1f} kJ/(kg*K). "
                f"Mass={m.mass:.0f} kg. Mushy zone width={2*m.dT_pc:.0f} degC. "
                f"UA_htf={m.UA_htf:.0f} W/K."
            ),
            "inputs": {
                "T_htf_in_degC":  {"unit": "degC", "range": [20.0, 90.0]},
                "flow_rate_kg_s": {"unit": "kg/s", "range": [0.0, 10.0]},
                "mode":           {"values": ["charge", "discharge", "idle"]},
                "duration_s":     {"unit": "s", "range": [0.0, 86400.0], "default": 3600.0},
                "T_ambient_degC": {"unit": "degC", "range": [-10.0, 50.0], "default": 25.0},
                "T_pcm_init":     {"unit": "degC", "optional": True},
            },
            "outputs": {
                "T_pcm_degC":        {"unit": "degC"},
                "phase_fraction":    {"unit": "dimensionless", "range": [0.0, 1.0]},
                "energy_stored_kwh": {"unit": "kWh"},
                "thermal_power_kw":  {"unit": "kW"},
                "T_outlet_degC":     {"unit": "degC"},
            },
            "source": "Mehling & Cabeza (2008); Voller (1990); Rubitherm RT58 datasheet",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    info = model.get_info()
    print(f"{info['ec_id']} -- {info['name']} -- {info['fidelity']}")

    r = model.predict({
        "T_htf_in_degC": 80.0, "flow_rate_kg_s": 1.0,
        "mode": "charge", "duration_s": 3600.0,
        "T_pcm_init": 40.0,
    })
    print(f"\nCharge 1h from 40 degC (HTF=80 degC, 1 kg/s):")
    for k, v in r.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
