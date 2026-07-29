"""EC126 — Flywheel Energy Storage — F1a Kinetic Model — Standardized Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import FlywheelF1a


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = FlywheelF1a(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        inputs:
            speed_rpm         : float or array, rotor speed [rpm, 8000-16000]
            torque_nm         : float or array, shaft torque [N·m, -100 to 100] (optional, default 0)
            time_hours        : float or array, standby time for self-discharge [h] (optional, default 0)
        returns:
            energy_stored_kwh : kinetic energy stored [kWh]
            soc               : state of charge [0-1]
            power_kw          : electrical power at terminals [kW] (+ charge, - discharge)
            self_discharge_kw : self-discharge power loss [kW]
            round_trip_efficiency : RTE including standby losses [-]
        """
        rpm = np.asarray(inputs["speed_rpm"], dtype=float)
        torque = np.asarray(inputs.get("torque_nm", 0.0), dtype=float)
        t_h = np.asarray(inputs.get("time_hours", 0.0), dtype=float)
        return {
            "energy_stored_kwh": self._model.energy_stored(rpm),
            "soc": self._model.soc(rpm),
            "power_kw": self._model.electrical_power(rpm, torque),
            "self_discharge_kw": self._model.self_discharge_power(rpm),
            "round_trip_efficiency": self._model.round_trip_efficiency(t_h),
        }

    def get_info(self) -> dict:
        return {
            "name": "Flywheel Energy Storage System",
            "ec_id": "EC126",
            "fidelity": "F1a",
            "model": "Kinetic Model",
            "description": "E=0.5*J*omega^2; SOC=(omega^2-omega_min^2)/(omega_max^2-omega_min^2)",
            "inputs": {
                "speed_rpm": {"unit": "rpm", "range": [8000.0, 16000.0]},
                "torque_nm": {"unit": "N·m", "range": [-100.0, 100.0], "default": 0.0},
                "time_hours": {"unit": "h", "range": [0.0, 24.0], "default": 0.0},
            },
            "outputs": {
                "energy_stored_kwh": {"unit": "kWh"},
                "soc": {"unit": "dimensionless"},
                "power_kw": {"unit": "kW"},
                "self_discharge_kw": {"unit": "kW"},
                "round_trip_efficiency": {"unit": "dimensionless"},
            },
            "source": "Arani et al. (2017), Energies, 10, 1361",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    print("At max speed (16000 rpm), full charge:")
    r = model.predict({"speed_rpm": 16000.0, "torque_nm": 0.0, "time_hours": 0.0})
    print(f"  Energy stored : {float(r['energy_stored_kwh']):.2f} kWh")
    print(f"  SOC           : {float(r['soc']):.3f}")
    print(f"  Self-discharge: {float(r['self_discharge_kw']):.3f} kW")
    print(f"  RTE (0h)      : {float(r['round_trip_efficiency'])*100:.1f} %")
    print()
    print("Discharging at 100 N·m, 12000 rpm:")
    r2 = model.predict({"speed_rpm": 12000.0, "torque_nm": -100.0})
    print(f"  Power (electrical): {float(r2['power_kw']):.1f} kW")
    print(f"  SOC              : {float(r2['soc']):.3f}")
