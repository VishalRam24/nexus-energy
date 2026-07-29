"""EC177 — BLDC Motor — F1b Thermal — Standardized Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import BLDCMotorF1b


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = BLDCMotorF1b(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        inputs:
            torque_nm           : float or array [Nm]
            speed_rpm           : float or array [rpm]
            magnet_temperature  : float or array [degC] (default 80)
            winding_temperature : float or array [degC] (default 80)
            ambient_temperature : float or array [degC] (default 25)
        returns:
            efficiency          : dimensionless
            input_power_w       : W
            output_power_w      : W
            losses_w            : W
            p_copper_w          : W
            p_iron_w            : W
            p_mech_w            : W
            p_stray_w           : W
            phase_current_A     : A
            derating_factor     : dimensionless
            demagnetization_risk: bool
        """
        torque = np.asarray(inputs["torque_nm"], dtype=float)
        speed = np.asarray(inputs["speed_rpm"], dtype=float)
        T_m = inputs.get("magnet_temperature", 80.0)
        T_w = inputs.get("winding_temperature", 80.0)
        T_a = inputs.get("ambient_temperature", 25.0)

        loss_dict = self._model.losses(torque, speed, T_m, T_w)
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
            "phase_current_A": self._model.phase_current(torque, T_m),
            "derating_factor": self._model.derating_factor(T_a),
            "demagnetization_risk": self._model.demagnetization_risk(T_m),
        }

    def get_info(self) -> dict:
        u = self.params["unit"]
        return {
            "name": "Brushless DC Motor (BLDC) Thermal",
            "ec_id": "EC177",
            "fidelity": "F1b",
            "description": (
                "Loss-split model: copper (temp-dep R_s, demagnetization-dep k_t), "
                "iron (speed^1.5), mechanical (speed), stray (output fraction). "
                "IEC 60034-1 ambient derating above 40 C."
            ),
            "inputs": {
                "torque_nm": {"unit": "Nm", "range": [0.0, 4.8]},
                "speed_rpm": {"unit": "rpm", "range": [0.0, 6000.0]},
                "magnet_temperature": {"unit": "degC", "range": [-20, 160], "default": 80},
                "winding_temperature": {"unit": "degC", "range": [-20, 160], "default": 80},
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
                "demagnetization_risk": {"unit": "bool"},
            },
            "params": {
                "P_rated": f"{u['P_rated_W']['value']} W",
                "R_s_ref": f"{u['R_s_ref']['value']} ohm @ {u['T_ref']['value']} C",
                "alpha_Cu": f"{u['alpha_Cu']['value']} 1/K",
                "alpha_Br": f"{u['alpha_Br']['value']} 1/K (NdFeB PM)",
            },
            "source": "Hanselman (2006); Gieras (2010); IEC 60034-30-1",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    for T_m in [25, 80, 120, 149]:
        r = model.predict({"torque_nm": 3.18, "speed_rpm": 3000.0,
                           "magnet_temperature": T_m, "winding_temperature": T_m})
        print(
            f"T_magnet={T_m:3d}C: eta={float(r['efficiency']):.4f}  "
            f"P_in={float(r['input_power_w']):.1f} W  "
            f"I={float(r['phase_current_A']):.2f} A  "
            f"demag={bool(r['demagnetization_risk'])}"
        )
