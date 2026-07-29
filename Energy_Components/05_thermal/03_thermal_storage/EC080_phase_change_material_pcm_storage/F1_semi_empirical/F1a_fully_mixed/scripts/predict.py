"""EC080 — PCM Storage — F1a Latent Heat Model — Standardized Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import PCMF1a


class ComponentModel:
    """Standardized interface for EC080 PCM Storage — F1a three-region latent heat model."""

    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = PCMF1a(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        Instantaneous rate-of-change and energy state for PCM storage.

        Args:
            inputs: {
                "temperature":      degC (0-80)   — current PCM temperature
                "liquid_fraction":  dimensionless (0-1) — current liquid fraction
                "q_charge":         W (0-10000)   — thermal input power
                "q_discharge":      W (0-10000)   — thermal output power
            }
        Returns:
            dict with:
                dT_dt           [K/s]
                d_fraction_dt   [1/s]
                energy_stored_kwh [kWh]
                soc             [-]
        """
        T  = np.asarray(inputs["temperature"],     dtype=float)
        f  = np.asarray(inputs["liquid_fraction"], dtype=float)
        Qc = np.asarray(inputs["q_charge"],        dtype=float)
        Qd = np.asarray(inputs["q_discharge"],     dtype=float)
        T_a = np.asarray(inputs.get("t_ambient", self._model.T_amb_ref), dtype=float)

        return {
            "dT_dt":             self._model.dT_dt(T, f, Qc, Qd, T_a),
            "d_fraction_dt":     self._model.df_dt(T, f, Qc, Qd, T_a),
            "energy_stored_kwh": self._model.energy_stored_kwh(T, f),
            "soc":               self._model.soc(T, f),
        }

    def get_info(self) -> dict:
        m = self._model
        return {
            "name":        "Phase-Change Material (PCM) Thermal Energy Storage",
            "ec_id":       "EC080",
            "fidelity":    "F1a",
            "description": (
                "Three-region latent heat model: solid (T<Tm), mushy (T=Tm, 0<f<1), liquid (T>Tm). "
                f"Paraffin RT42: Tm={m.Tm}°C, L=174 kJ/kg, cp=2.0 kJ/(kg·K), mass={m.mass}kg."
            ),
            "inputs": {
                "temperature":     {"unit": "degC",          "range": [0.0,    80.0]},
                "liquid_fraction": {"unit": "dimensionless", "range": [0.0,     1.0]},
                "q_charge":        {"unit": "W",             "range": [0.0, 10000.0]},
                "q_discharge":     {"unit": "W",             "range": [0.0, 10000.0]},
            },
            "outputs": {
                "dT_dt":             {"unit": "K/s"},
                "d_fraction_dt":     {"unit": "1/s"},
                "energy_stored_kwh": {"unit": "kWh"},
                "soc":               {"unit": "dimensionless"},
            },
            "source": (
                "Mehling & Cabeza (2008), Heat and Cold Storage with PCM, Springer; "
                "Rubitherm Technologies RT42 datasheet"
            ),
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    print(json.dumps(model.get_info(), indent=2))
    # Test all three regions
    for T, f, label in [(20, 0.0, "Solid"), (42, 0.5, "Mushy"), (60, 1.0, "Liquid")]:
        r = model.predict({"temperature": float(T), "liquid_fraction": f,
                           "q_charge": 500.0, "q_discharge": 0.0})
        print(f"\n{label} region (T={T}°C, f={f}):")
        for k, v in r.items():
            print(f"  {k}: {float(v):.6f}")
