"""EC122 — Pumped Hydro Storage — F1a Round-Trip — Standardized Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import PHSF1a


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = PHSF1a(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        inputs:
            mode         : str  — "generate" or "pump"
            flow_rate    : float or array [m3/s]
            head         : float or array [m]
        returns:
            power_kw             : float or array [kW]
            efficiency           : float  (one-way, dimensionless)
            energy_capacity_gwh  : float  [GWh] for default reservoir at given head
            round_trip_eta       : float  (dimensionless)
        """
        mode = inputs.get("mode", "generate").strip().lower()
        Q = np.asarray(inputs["flow_rate"], dtype=float)
        H = np.asarray(inputs["head"], dtype=float)

        if mode == "generate":
            power = self._model.generation_power(Q, H)
            eff = self._model.generation_efficiency()
        elif mode == "pump":
            power = self._model.pumping_power(Q, H)
            eff = self._model.pump_efficiency()
        else:
            raise ValueError(f"mode must be 'generate' or 'pump', got '{mode}'")

        energy_gwh = self._model.energy_capacity(H)
        rte = self._model.round_trip_efficiency()

        return {
            "power_kw": power,
            "efficiency": float(eff),
            "energy_capacity_gwh": energy_gwh,
            "round_trip_eta": float(rte),
        }

    def get_info(self) -> dict:
        return {
            "name": "Pumped Hydro Storage (PHS)",
            "ec_id": "EC122",
            "fidelity": "F1a",
            "description": (
                "Round-trip model: P_gen = eta_t*eta_g*rho*g*Q*H/1000 [kW]; "
                "P_pump = rho*g*Q*H/(eta_p*eta_m*1000) [kW]; "
                "RTE = eta_t*eta_g*eta_p*eta_m"
            ),
            "inputs": {
                "mode": {"unit": "-", "values": ["generate", "pump"]},
                "flow_rate": {"unit": "m3/s", "range": [1.0, 500.0]},
                "head": {"unit": "m", "range": [10.0, 1000.0]},
            },
            "outputs": {
                "power_kw": {"unit": "kW"},
                "efficiency": {"unit": "dimensionless"},
                "energy_capacity_gwh": {"unit": "GWh"},
                "round_trip_eta": {"unit": "dimensionless"},
            },
            "source": "Rehman et al. (2015), Renew. Sustain. Energy Rev., 44, 586-598",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    r_gen = model.predict({"mode": "generate", "flow_rate": 50.0, "head": 300.0})
    r_pump = model.predict({"mode": "pump", "flow_rate": 50.0, "head": 300.0})
    print(f"Generation: P={float(r_gen['power_kw']):.1f} kW, eta={r_gen['efficiency']:.3f}")
    print(f"Pumping:    P={float(r_pump['power_kw']):.1f} kW, eta={r_pump['efficiency']:.3f}")
    print(f"Round-trip eta: {r_gen['round_trip_eta']:.3f}")
    print(f"Energy capacity: {float(r_gen['energy_capacity_gwh']):.3f} GWh")
