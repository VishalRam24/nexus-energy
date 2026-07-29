"""EC079 — Molten Salt TES — F1a Fully Mixed — Standardized Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import MoltenSaltTESF1a


class ComponentModel:
    """Standardized interface for EC079 Molten Salt TES — F1a fully mixed model."""

    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = MoltenSaltTESF1a(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        Steady-state / instantaneous evaluation of storage state and rate-of-change.

        Args:
            inputs: {
                "temperature":   degC (220-580) — current salt temperature
                "q_charge":      kW (0-100000)  — thermal input power
                "q_discharge":   kW (0-100000)  — thermal output power
                "t_ambient":     degC (-20 to 50), default=25
            }
        Returns:
            dict with:
                dT_dt            [K/s]    — temperature rate of change
                energy_stored_mwh [MWh]  — energy stored above cold reference
                soc              [-]      — state of charge [0,1]
                heat_loss_kw     [kW]    — instantaneous heat loss to environment
        """
        T   = np.asarray(inputs["temperature"], dtype=float)
        Qc  = np.asarray(inputs["q_charge"],    dtype=float)
        Qd  = np.asarray(inputs["q_discharge"], dtype=float)
        T_a = np.asarray(inputs.get("t_ambient", self._model.T_amb_ref), dtype=float)

        return {
            "dT_dt":             self._model.dT_dt(T, Qc, Qd, T_a),
            "energy_stored_mwh": self._model.energy_stored_mwh(T),
            "soc":               self._model.soc(T),
            "heat_loss_kw":      self._model.heat_loss_kw(T, T_a),
        }

    def get_info(self) -> dict:
        m = self._model
        return {
            "name":        "Molten Salt Thermal Energy Storage",
            "ec_id":       "EC079",
            "fidelity":    "F1a",
            "description": (
                "Fully mixed (0D) energy balance: dT/dt = (Q_charge - Q_discharge - UA*(T-T_amb)) / (m*cp). "
                f"Solar salt 60% NaNO3 + 40% KNO3, 1000 m³, {m.E_capacity_MWh:.1f} MWh capacity."
            ),
            "inputs": {
                "temperature":  {"unit": "degC", "range": [220.0, 580.0]},
                "q_charge":     {"unit": "kW",   "range": [0.0, 100000.0]},
                "q_discharge":  {"unit": "kW",   "range": [0.0, 100000.0]},
                "t_ambient":    {"unit": "degC", "range": [-20.0, 50.0], "default": 25.0},
            },
            "outputs": {
                "dT_dt":             {"unit": "K/s"},
                "energy_stored_mwh": {"unit": "MWh"},
                "soc":               {"unit": "dimensionless"},
                "heat_loss_kw":      {"unit": "kW"},
            },
            "source": "Herrmann et al. (2004), Energy 29(5-6), 883-893",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    info = model.get_info()
    print(f"E_capacity: {model._model.E_capacity_MWh:.1f} MWh")
    r = model.predict({"temperature": 427.5, "q_charge": 50000.0, "q_discharge": 0.0})
    print(f"\nMid-SOC, charging at 50 MW:")
    for k, v in r.items():
        print(f"  {k}: {float(v):.4f}")
