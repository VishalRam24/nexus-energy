"""EC178 — SRM — F1b Thermal — Standardized Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import SRMMotorF1b


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = SRMMotorF1b(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        inputs:
            torque_nm           : float or array [Nm]
            speed_rpm           : float or array [rpm]
            winding_temperature : float or array [degC] (default 75)
            ambient_temperature : float or array [degC] (default 25)
        returns:
            efficiency, input_power_w, output_power_w, losses_w,
            p_copper_w, p_iron_w, p_mech_w, p_stray_w,
            phase_current_A, derating_factor
        """
        torque = np.asarray(inputs["torque_nm"], dtype=float)
        speed = np.asarray(inputs["speed_rpm"], dtype=float)
        T_w = inputs.get("winding_temperature", 75.0)
        T_a = inputs.get("ambient_temperature", 25.0)

        loss_dict = self._model.losses(torque, speed, T_w)
        P_out = self._model.output_power(torque, speed)
        P_in = P_out + loss_dict["p_total_w"]
        eta = np.where(P_in > 1e-6, P_out / P_in, 0.0)
        eta = np.clip(eta, 0.0, 1.0)

        return {
            "efficiency": eta,
            "input_power_w": P_in,
            "output_power_w": P_out,
            "losses_w": loss_dict["p_total_w"],
            "p_copper_w": loss_dict["p_copper_w"],
            "p_iron_w": loss_dict["p_iron_w"],
            "p_mech_w": loss_dict["p_mech_w"],
            "p_stray_w": loss_dict["p_stray_w"],
            "phase_current_A": self._model.phase_current(torque, speed),
            "derating_factor": self._model.derating_factor(T_a),
        }

    def get_info(self) -> dict:
        u = self.params["unit"]
        return {
            "name": "Switched Reluctance Motor (SRM) Thermal",
            "ec_id": "EC178",
            "fidelity": "F1b",
            "description": (
                "Loss breakdown: copper (T-dep R_ph), iron (eddy omega^2 + hyst omega), "
                "mechanical (omega), stray (P_out fraction). No PMs — no demagnetization. "
                "IEC 60034-1 ambient derating above 40 C."
            ),
            "inputs": {
                "torque_nm": {"unit": "Nm", "range": [0.0, 15.0]},
                "speed_rpm": {"unit": "rpm", "range": [0.0, 9000.0]},
                "winding_temperature": {"unit": "degC", "range": [-20, 180], "default": 75},
                "ambient_temperature": {"unit": "degC", "range": [-20, 60], "default": 25},
            },
            "outputs": {
                "efficiency": {"unit": "dimensionless"},
                "input_power_w": {"unit": "W"},
                "output_power_w": {"unit": "W"},
                "losses_w": {"unit": "W"},
                "p_copper_w": {"unit": "W"},
                "p_iron_w": {"unit": "W"},
                "p_mech_w": {"unit": "W"},
                "p_stray_w": {"unit": "W"},
                "phase_current_A": {"unit": "A"},
                "derating_factor": {"unit": "dimensionless"},
            },
            "params": {
                "P_rated": f"{u['P_rated_W']['value']} W",
                "R_ph_ref": f"{u['R_ph_ref']['value']} ohm @ {u['T_ref']['value']} C",
                "alpha_Cu": f"{u['alpha_Cu']['value']} 1/K",
                "n_phases": str(u['n_phases']['value']),
            },
            "source": "Miller (1993); Krishnan (2001); IEC 60034-30-1",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    for T_w in [25, 75, 120, 155]:
        r = model.predict({"torque_nm": 9.55, "speed_rpm": 3000.0,
                           "winding_temperature": T_w})
        print(
            f"T_winding={T_w:3d}C: eta={float(r['efficiency']):.4f}  "
            f"P_in={float(r['input_power_w']):.1f} W  "
            f"P_copper={float(r['p_copper_w']):.1f} W"
        )
