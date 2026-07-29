"""EC127 — Gravity Energy Storage — F1a Potential Energy Model — Standardized Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import GravityF1a


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = GravityF1a(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        inputs:
            mode             : str — "charge", "discharge", or "idle"
            velocity_mps     : float or array, lift / lower speed [m/s] (optional)
            soc              : float or array, current SOC [-] (optional, default 0.5)
        returns:
            power_kw              : electrical power at terminals [kW] (+charge, -discharge)
            height_m              : mass height at given SOC [m]
            potential_energy_kwh  : stored potential energy [kWh]
            energy_capacity_kwh   : maximum usable capacity [kWh]
            round_trip_eta        : round-trip electrical efficiency [-]
            charge_eta            : one-way charge efficiency [-]
            discharge_eta         : one-way discharge efficiency [-]
        """
        mode = inputs.get("mode", "idle").strip().lower()
        v = np.asarray(inputs.get("velocity_mps", 0.0), dtype=float)
        soc = np.asarray(inputs.get("soc", 0.5), dtype=float)

        if mode == "charge":
            power = self._model.charge_power(v)             # +kW
        elif mode == "discharge":
            power = -self._model.discharge_power(v)         # -kW (delivered)
        elif mode == "idle":
            power = np.zeros_like(v)
        else:
            raise ValueError(f"mode must be 'charge', 'discharge', or 'idle', got '{mode}'")

        return {
            "power_kw": power,
            "height_m": self._model.height(soc),
            "potential_energy_kwh": self._model.potential_energy_kwh(soc),
            "energy_capacity_kwh": float(self._model.energy_capacity_kwh()),
            "round_trip_eta": float(self._model.round_trip_efficiency()),
            "charge_eta": float(self._model.charge_efficiency()),
            "discharge_eta": float(self._model.discharge_efficiency()),
        }

    def get_info(self) -> dict:
        return {
            "name": "Gravity Energy Storage",
            "ec_id": "EC127",
            "fidelity": "F1a",
            "description": (
                "Potential energy model: E = m*g*h; "
                "P_charge = m*g*v/(eta_m*eta_d); "
                "P_discharge = m*g*v*eta_d*eta_g; "
                "RTE = eta_m*eta_d^2*eta_g"
            ),
            "inputs": {
                "mode": {"unit": "-", "values": ["charge", "discharge", "idle"]},
                "velocity_mps": {"unit": "m/s", "range": [0.0, 5.0]},
                "soc": {"unit": "-", "range": [0.0, 1.0]},
            },
            "outputs": {
                "power_kw": {"unit": "kW"},
                "height_m": {"unit": "m"},
                "potential_energy_kwh": {"unit": "kWh"},
                "energy_capacity_kwh": {"unit": "kWh"},
                "round_trip_eta": {"unit": "dimensionless"},
                "charge_eta": {"unit": "dimensionless"},
                "discharge_eta": {"unit": "dimensionless"},
            },
            "source": "Botha & Kamper (2019), J. Energy Storage, 23, 159-174; Berrada et al. (2017), Energy Conversion and Management, 137, 191-200",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    r_c = model.predict({"mode": "charge", "velocity_mps": 0.05, "soc": 0.5})
    r_d = model.predict({"mode": "discharge", "velocity_mps": 0.05, "soc": 0.5})
    print(f"Charge P_in    : {float(r_c['power_kw']):.1f} kW (v=0.05 m/s)")
    print(f"Discharge P_out: {-float(r_d['power_kw']):.1f} kW (v=0.05 m/s)")
    print(f"Height @SOC=0.5: {float(r_c['height_m']):.1f} m")
    print(f"Potential E    : {float(r_c['potential_energy_kwh']):.2f} kWh")
    print(f"Energy capacity: {r_c['energy_capacity_kwh']/1000:.2f} MWh")
    print(f"Round-trip eta : {r_c['round_trip_eta']*100:.2f} %")
