"""EC180 — DFIG — F1b Thermal — Standardized Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import DFIGF1b


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = DFIGF1b(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        inputs:
            load_fraction       : float or array [0.05, 1.2]
            slip                : float or array [-0.35, 0.35] (default rated_slip=-0.25)
            stator_temperature  : float or array [degC] (default 75)
            rotor_temperature   : float or array [degC] (default 75)
            ambient_temperature : float or array [degC] (default 25)
        returns:
            efficiency, input_power_w, output_power_w, losses_w,
            p_stator_cu_w, p_rotor_cu_w, p_iron_w, p_mech_w, p_stray_w,
            p_converter_w, stator_current_A, rotor_speed_rpm, derating_factor
        """
        plr = np.asarray(inputs["load_fraction"], dtype=float)
        slip = inputs.get("slip", None)
        T_s = inputs.get("stator_temperature", 75.0)
        T_r = inputs.get("rotor_temperature", 75.0)
        T_a = inputs.get("ambient_temperature", 25.0)

        loss_dict = self._model.losses(plr, slip, T_s, T_r)
        P_out = self._model.output_power(plr)
        P_in = P_out + loss_dict["p_total_w"]
        eta = np.where(P_in > 1.0, P_out / P_in, 0.0)
        eta = np.clip(eta, 0.0, 1.0)

        s = slip if slip is not None else self._model.slip_rated

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
            "p_converter_w": loss_dict["p_converter_w"],
            "stator_current_A": self._model.stator_current(plr),
            "rotor_speed_rpm": self._model.rotor_speed_rpm(s),
            "derating_factor": self._model.derating_factor(T_a),
        }

    def get_info(self) -> dict:
        u = self.params["unit"]
        return {
            "name": "Doubly-Fed Induction Generator (DFIG) Thermal",
            "ec_id": "EC180",
            "fidelity": "F1b",
            "description": (
                "Loss breakdown: stator Cu (T-dep), rotor Cu (T-dep), "
                "iron (const), mechanical (const), stray, converter (rotor circuit). "
                "Slip sets rotor speed (negative = super-synchronous). "
                "IEC 60034-1 derating above 40 C."
            ),
            "inputs": {
                "load_fraction": {"unit": "dimensionless", "range": [0.05, 1.2]},
                "slip": {"unit": "dimensionless", "range": [-0.35, 0.35], "default": -0.25},
                "stator_temperature": {"unit": "degC", "range": [-20, 180], "default": 75},
                "rotor_temperature": {"unit": "degC", "range": [-20, 180], "default": 75},
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
                "p_converter_w": {"unit": "W"},
                "stator_current_A": {"unit": "A"},
                "rotor_speed_rpm": {"unit": "rpm"},
                "derating_factor": {"unit": "dimensionless"},
            },
            "params": {
                "P_rated": f"{u['P_rated_MW']['value']} MW",
                "V_stator": f"{u['V_stator_kV']['value']} kV",
                "eta_converter": str(u["eta_converter"]["value"]),
                "alpha_Cu": f"{u['alpha_Cu']['value']} 1/K",
            },
            "source": "Muller et al. (2002); Boldea (2006); IEC 60034-30-1",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    for T in [25, 75, 120, 155]:
        r = model.predict({"load_fraction": 1.0, "slip": -0.25,
                           "stator_temperature": T, "rotor_temperature": T})
        print(
            f"T={T:3d}C slip=-0.25: eta={float(r['efficiency']):.4f}  "
            f"P_in={float(r['input_power_w'])/1e6:.3f} MW  "
            f"P_conv={float(r['p_converter_w'])/1e3:.1f} kW"
        )
