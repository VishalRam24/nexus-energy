"""EC126 — Flywheel Energy Storage — F1b Thermal — Standardized Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import FlywheelF1b


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = FlywheelF1b(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        inputs:
            soc                 : float or array [0-1]
            power_command_kw    : float or array [kW] (+charge, -discharge)
            ambient_temperature : float or array [degC] (default 25)
        returns:
            power_actual_kw, losses_kw, self_discharge_rate_per_hour,
            efficiency, speed_rpm
        """
        soc = np.asarray(inputs["soc"], dtype=float)
        P_cmd = np.asarray(inputs.get("power_command_kw", 0.0), dtype=float)
        T_amb = inputs.get("ambient_temperature", 25.0)

        return {
            "power_actual_kw": self._model.power_actual(soc, P_cmd, T_amb),
            "losses_kw": self._model.losses(soc, P_cmd, T_amb),
            "self_discharge_rate_per_hour": self._model.self_discharge_rate(soc, T_amb),
            "efficiency": self._model.efficiency(soc, P_cmd, T_amb),
            "speed_rpm": self._model.speed_rpm(soc),
        }

    def get_info(self) -> dict:
        u = self.params["unit"]
        return {
            "name": "Flywheel Energy Storage (Thermal)",
            "ec_id": "EC126",
            "fidelity": "F1b",
            "description": (
                "Speed-dependent losses: P_windage=k_w*omega^3 (cubic), "
                "P_bearing=k_b*omega (linear); air density varies with temperature"
            ),
            "inputs": {
                "soc": {"unit": "dimensionless", "range": [0.0, 1.0]},
                "power_command_kw": {"unit": "kW", "range": [-100, 100]},
                "ambient_temperature": {"unit": "degC", "range": [-20, 60], "default": 25},
            },
            "outputs": {
                "power_actual_kw": {"unit": "kW"},
                "losses_kw": {"unit": "kW"},
                "self_discharge_rate_per_hour": {"unit": "1/h"},
                "efficiency": {"unit": "dimensionless"},
                "speed_rpm": {"unit": "rpm"},
            },
            "params": {
                "J": f"{u['J_kgm2']['value']} kg*m2",
                "omega_max": f"{u['omega_max_rpm']['value']} rpm",
                "E_max": f"{u['E_max_kwh']['value']} kWh",
                "k_windage": u["k_windage"]["value"],
                "k_bearing": u["k_bearing"]["value"],
            },
            "source": "Arani et al. (2017); Beacon Power; Genta (2005)",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    for soc in [0.1, 0.25, 0.5, 0.75, 1.0]:
        r = model.predict({"soc": soc, "power_command_kw": -50.0})
        print(
            f"SOC={soc:.2f}: P_actual={float(r['power_actual_kw']):.2f}kW  "
            f"loss={float(r['losses_kw']):.3f}kW  "
            f"sd_rate={float(r['self_discharge_rate_per_hour']):.4f}/h  "
            f"rpm={float(r['speed_rpm']):.0f}"
        )
