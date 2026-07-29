"""EC176 — PMSM — F1b Thermal — Standardized Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import PMSMF1b


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = PMSMF1b(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        inputs:
            torque            : Nm [0-25]
            speed_rpm         : rpm [0-12000]
            magnet_temperature: degC (default 80)
            ambient_temperature: degC (default 25)
        returns:
            efficiency, output_power_kw, input_power_kw, total_losses_kw,
            torque_Nm (echo), back_emf_V, derating_factor, demag_risk
        """
        T = np.asarray(inputs["torque"], dtype=float)
        omega = np.asarray(inputs["speed_rpm"], dtype=float)
        T_mag = inputs.get("magnet_temperature", 80.0)
        T_amb = inputs.get("ambient_temperature", 25.0)

        p_out = self._model.output_power(T, omega)
        p_in = self._model.input_power(T, omega, T_mag)
        losses = self._model.losses(T, omega, T_mag)

        return {
            "efficiency": self._model.efficiency(T, omega, T_mag),
            "output_power_kw": p_out / 1000.0,
            "input_power_kw": p_in / 1000.0,
            "total_losses_kw": losses["p_total_w"] / 1000.0,
            "torque_Nm": T,
            "back_emf_V": self._model.back_emf(omega, T_mag),
            "derating_factor": self._model.derating_factor(T_mag, T_amb),
            "demag_risk": self._model.demagnetization_risk(T_mag),
        }

    def get_info(self) -> dict:
        return {
            "name": "PMSM (Thermal)",
            "ec_id": "EC176",
            "fidelity": "F1b",
            "description": (
                "PM flux demagnetization: Phi_m(T)=Phi_m_ref*(1+alpha_Br*(T-T_ref)); "
                "R_s(T) copper temp dependence; NdFeB demag risk above 150C"
            ),
            "inputs": {
                "torque": {"unit": "Nm", "range": [0, 25]},
                "speed_rpm": {"unit": "rpm", "range": [0, 12000]},
                "magnet_temperature": {"unit": "degC", "range": [20, 180], "default": 80},
                "ambient_temperature": {"unit": "degC", "range": [-20, 60], "default": 25},
            },
            "outputs": {
                "efficiency": {"unit": "dimensionless"},
                "output_power_kw": {"unit": "kW"},
                "input_power_kw": {"unit": "kW"},
                "total_losses_kw": {"unit": "kW"},
                "torque_Nm": {"unit": "Nm"},
                "back_emf_V": {"unit": "V"},
                "derating_factor": {"unit": "dimensionless"},
                "demag_risk": {"unit": "boolean"},
            },
            "source": "Gieras (2010); Arnold Magnetic Technologies NdFeB data",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    for T_mag in [25, 60, 80, 120, 150, 170]:
        r = model.predict({"torque": 16.0, "speed_rpm": 3000.0,
                           "magnet_temperature": T_mag})
        print(
            f"T_mag={T_mag:>3}C: eta={float(r['efficiency']):.4f}  "
            f"P_out={float(r['output_power_kw']):.2f}kW  "
            f"back_EMF={float(r['back_emf_V']):.1f}V  "
            f"demag={bool(r['demag_risk'])}"
        )
