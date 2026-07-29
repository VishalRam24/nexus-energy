"""EC179 — WRSG — F1b Thermal — Standardized Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import WRSGF1b


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = WRSGF1b(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        inputs:
            load_fraction        : float or array [0.05, 1.2]
            stator_temperature   : float or array [degC] (default 75)
            rotor_temperature    : float or array [degC] (default 75)
            power_factor         : float or array (default rated pf=0.9)
            ambient_temperature  : float or array [degC] (default 25)
        returns:
            efficiency, input_power_w, output_power_w, losses_w,
            p_stator_cu_w, p_rotor_cu_w, p_iron_w, p_mech_w, p_stray_w,
            stator_current_A, field_current_A, derating_factor
        """
        plr = np.asarray(inputs["load_fraction"], dtype=float)
        T_s = inputs.get("stator_temperature", 75.0)
        T_r = inputs.get("rotor_temperature", 75.0)
        pf = inputs.get("power_factor", None)
        T_a = inputs.get("ambient_temperature", 25.0)

        loss_dict = self._model.losses(plr, T_s, T_r, pf)
        P_out = self._model.output_power(plr)
        P_in = P_out + loss_dict["p_total_w"]
        eta = np.where(P_in > 1.0, P_out / P_in, 0.0)
        eta = np.clip(eta, 0.0, 1.0)

        return {
            "efficiency": eta,
            "input_power_w": P_in,
            "output_power_w": P_out,
            "losses_w": loss_dict["p_total_w"],
            "p_stator_cu_w": loss_dict["p_stator_cu_w"],
            "p_rotor_cu_w": loss_dict["p_rotor_cu_w"],
            "p_iron_w": loss_dict["p_iron_w"],
            "p_mech_w": loss_dict["p_mech_w"],
            "p_stray_w": loss_dict["p_stray_w"],
            "stator_current_A": self._model.stator_current(plr, pf),
            "field_current_A": self._model.field_current(plr, pf),
            "derating_factor": self._model.derating_factor(T_a),
        }

    def get_info(self) -> dict:
        u = self.params["unit"]
        return {
            "name": "Wound Rotor Synchronous Generator Thermal",
            "ec_id": "EC179",
            "fidelity": "F1b",
            "description": (
                "Loss breakdown: stator Cu (T-dep R_s), rotor Cu (T-dep R_f), "
                "iron (const at sync speed), mechanical (const), stray. "
                "IEC 60034-1 derating above 40 C ambient."
            ),
            "inputs": {
                "load_fraction": {"unit": "dimensionless", "range": [0.05, 1.2]},
                "stator_temperature": {"unit": "degC", "range": [-20, 180], "default": 75},
                "rotor_temperature": {"unit": "degC", "range": [-20, 180], "default": 75},
                "power_factor": {"unit": "dimensionless", "range": [0.5, 1.0], "default": 0.9},
                "ambient_temperature": {"unit": "degC", "range": [-20, 60], "default": 25},
            },
            "outputs": {
                "efficiency": {"unit": "dimensionless"},
                "input_power_w": {"unit": "W"},
                "output_power_w": {"unit": "W"},
                "losses_w": {"unit": "W"},
                "p_stator_cu_w": {"unit": "W"},
                "p_rotor_cu_w": {"unit": "W"},
                "p_iron_w": {"unit": "W"},
                "p_mech_w": {"unit": "W"},
                "p_stray_w": {"unit": "W"},
                "stator_current_A": {"unit": "A"},
                "field_current_A": {"unit": "A"},
                "derating_factor": {"unit": "dimensionless"},
            },
            "params": {
                "S_rated": f"{u['S_rated_kVA']['value']} kVA",
                "V_line": f"{u['V_line_kV']['value']} kV",
                "R_s_ref": f"{u['R_s_ref']['value']} ohm @ {u['T_ref']['value']} C",
                "R_f_ref": f"{u['R_f_ref']['value']} ohm @ {u['T_ref']['value']} C",
                "alpha_Cu": f"{u['alpha_Cu']['value']} 1/K",
            },
            "source": "Kundur (1994); Chapman (2012); IEC 60034-30-1",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    for T in [25, 75, 120, 155]:
        r = model.predict({"load_fraction": 1.0, "stator_temperature": T, "rotor_temperature": T})
        print(
            f"T={T:3d}C: eta={float(r['efficiency']):.4f}  "
            f"P_in={float(r['input_power_w'])/1000:.1f} kW  "
            f"P_stator_cu={float(r['p_stator_cu_w']):.0f} W  "
            f"P_rotor_cu={float(r['p_rotor_cu_w']):.0f} W"
        )
