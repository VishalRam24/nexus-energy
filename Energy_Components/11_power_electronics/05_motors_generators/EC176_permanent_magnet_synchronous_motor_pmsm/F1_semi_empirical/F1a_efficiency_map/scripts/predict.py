"""EC176 — PMSM — F1a Efficiency Map — Standardized Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import PMSMF1a


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = PMSMF1a(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        inputs:
            torque (Nm):     Output torque [0–200]
            speed_rpm (rpm): Rotor speed [0–12000]

        returns:
            efficiency (-)
            output_power_kw (kW)
            input_power_kw (kW)
            total_losses_kw (kW)
        """
        T = np.asarray(inputs["torque"], dtype=float)
        omega = np.asarray(inputs["speed_rpm"], dtype=float)
        p_out = self._model.output_power(T, omega)
        p_in = self._model.input_power(T, omega)
        losses = self._model.losses(T, omega)
        return {
            "efficiency": self._model.efficiency(T, omega),
            "output_power_kw": p_out / 1000.0,
            "input_power_kw": p_in / 1000.0,
            "total_losses_kw": losses["p_total_w"] / 1000.0,
        }

    def get_info(self) -> dict:
        return {
            "name": "Permanent Magnet Synchronous Motor (PMSM)",
            "ec_id": "EC176",
            "fidelity": "F1a",
            "description": (
                "Loss separation: P_copper=(T/k_t)^2*R_s, "
                "P_iron=k_e*omega^1.5, P_mech=k_f*omega; "
                "eta = P_out / (P_out + P_loss)"
            ),
            "inputs": {
                "torque": {"unit": "Nm", "range": [0.0, 200.0]},
                "speed_rpm": {"unit": "rpm", "range": [0.0, 12000.0]},
            },
            "outputs": {
                "efficiency": {"unit": "dimensionless"},
                "output_power_kw": {"unit": "kW"},
                "input_power_kw": {"unit": "kW"},
                "total_losses_kw": {"unit": "kW"},
            },
            "source": "Gieras (2010), Permanent Magnet Motor Technology, 3rd ed. CRC Press",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    # Rated operating point
    r = model.predict({"torque": 160.0, "speed_rpm": 3000.0})
    print(f"Rated point: eta={float(r['efficiency']):.4f}, "
          f"P_out={float(r['output_power_kw']):.2f}kW, "
          f"P_in={float(r['input_power_kw']):.2f}kW, "
          f"P_loss={float(r['total_losses_kw']):.3f}kW")
    # Peak efficiency search
    import numpy as np
    T_arr = np.linspace(10, 200, 50)
    omega_arr = np.linspace(500, 6000, 50)
    TT, OO = np.meshgrid(T_arr, omega_arr)
    r2 = model.predict({"torque": TT.ravel(), "speed_rpm": OO.ravel()})
    eta_peak = float(np.max(r2["efficiency"]))
    print(f"Peak efficiency in operating range: {eta_peak:.4f}")
