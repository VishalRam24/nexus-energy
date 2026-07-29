"""EC127 — Gravity Energy Storage — F1b Losses — Standardized Predict Interface"""
import json
import numpy as np
from pathlib import Path
from model import GravityF1b


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = GravityF1b(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        inputs:
            soc             : float or array [0-1]
            velocity_mps    : lift/lower velocity [m/s]
            plf             : part-load factor [0-1] (optional; computed if not given)
            mode            : 'charge' | 'discharge' | 'losses' (default 'discharge')
        returns (charge/discharge):
            power_kw, efficiency, round_trip_efficiency,
            friction_loss_kw, drag_loss_kw, bearing_loss_kw, total_mech_loss_kw,
            motor_efficiency, generator_efficiency, potential_energy_kwh
        returns (losses):
            friction_kw, drag_kw, bearing_kw, total_kw (breakdown by component)
        """
        m = self._model
        mode = inputs.get("mode", "discharge")
        soc = np.asarray(inputs.get("soc", 0.5), dtype=float)
        v = np.asarray(inputs.get("velocity_mps", 1.0), dtype=float)
        plf = inputs.get("plf", None)

        losses = m.mechanical_loss_breakdown(v)

        if mode == "losses":
            return {
                "friction_kw": losses["friction_kw"],
                "drag_kw": losses["drag_kw"],
                "bearing_kw": losses["bearing_kw"],
                "total_mech_loss_kw": losses["total_kw"],
                "velocity_mps": v,
            }

        if mode == "charge":
            P_kw = m.charge_power(v, plf)
        else:
            P_kw = m.discharge_power(v, plf)

        plf_v = plf
        if plf_v is None:
            P_grav = m.m * m.g * np.abs(v) / 1000.0
            P_loss = losses["total_kw"]
            if mode == "charge":
                P_shaft = P_grav + P_loss
            else:
                P_shaft = np.maximum(P_grav - P_loss, 0.0)
            plf_v = np.clip(P_shaft * 1000.0 / m.P_rated, 1e-6, 1.0)

        eta = m.efficiency(v, mode, plf_v)
        rte = m.round_trip_efficiency(v, plf_v)

        return {
            "power_kw": P_kw,
            "efficiency": eta,
            "round_trip_efficiency": rte,
            "friction_loss_kw": losses["friction_kw"],
            "drag_loss_kw": losses["drag_kw"],
            "bearing_loss_kw": losses["bearing_kw"],
            "total_mech_loss_kw": losses["total_kw"],
            "motor_efficiency": m.motor_efficiency(plf_v),
            "generator_efficiency": m.generator_efficiency(plf_v),
            "potential_energy_kwh": m.potential_energy_kwh(soc),
        }

    def get_info(self) -> dict:
        u = self.params["unit"]
        m = self._model
        return {
            "name": "Gravity Energy Storage (Losses Model)",
            "ec_id": "EC127",
            "fidelity": "F1b",
            "description": (
                "Adds speed-dependent mechanical losses (Coulomb friction + aerodynamic drag + "
                "bearing losses) and partial-load motor/generator efficiency curves."
            ),
            "inputs": {
                "soc": {"unit": "dimensionless", "range": [0.0, 1.0]},
                "velocity_mps": {"unit": "m/s", "range": [0.0, u["v_max_mps"]["value"]]},
                "plf": {"unit": "dimensionless", "range": [0.0, 1.0], "optional": True},
                "mode": {"values": ["charge", "discharge", "losses"]},
            },
            "outputs": {
                "power_kw": {"unit": "kW"},
                "efficiency": {"unit": "dimensionless"},
                "round_trip_efficiency": {"unit": "dimensionless"},
                "friction_loss_kw": {"unit": "kW"},
                "drag_loss_kw": {"unit": "kW"},
                "bearing_loss_kw": {"unit": "kW"},
                "total_mech_loss_kw": {"unit": "kW"},
                "motor_efficiency": {"unit": "dimensionless"},
                "generator_efficiency": {"unit": "dimensionless"},
            },
            "params": {
                "mass_kg": u["mass_kg"]["value"],
                "h_max_m": u["h_max_m"]["value"],
                "P_rated_kw": u["P_rated_kw"]["value"],
                "friction_coeff": u["friction_coeff"]["value"],
                "cable_drag_k": u["cable_drag_k"]["value"],
                "eta_motor_rated": u["eta_motor_rated"]["value"],
                "eta_gen_rated": u["eta_gen_rated"]["value"],
            },
            "source": "Botha & Kamper (2019); Berrada et al. (2017); Pyrhonen et al. (2013)",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    print("=== EC127 F1b Gravity Storage Losses Model ===\n")
    print("Losses vs velocity:")
    for v in [0.1, 0.5, 1.0, 2.0, 3.0]:
        r = model.predict({"velocity_mps": v, "mode": "losses"})
        print(f"  v={v:.1f} m/s: friction={float(r['friction_kw']):.2f}  "
              f"drag={float(r['drag_kw']):.4f}  "
              f"bearing={float(r['bearing_loss_kw']):.2f}  "
              f"total={float(r['total_mech_loss_kw']):.2f} kW")

    print("\nDischarge power vs velocity:")
    for v in [0.5, 1.0, 2.0, 3.0]:
        r = model.predict({"soc": 0.5, "velocity_mps": v, "mode": "discharge"})
        print(f"  v={v:.1f}: P_out={float(r['power_kw']):.0f} kW  "
              f"eta={float(r['efficiency']):.4f}  "
              f"RTE={float(r['round_trip_efficiency']):.4f}  "
              f"eta_gen={float(r['generator_efficiency']):.4f}")
