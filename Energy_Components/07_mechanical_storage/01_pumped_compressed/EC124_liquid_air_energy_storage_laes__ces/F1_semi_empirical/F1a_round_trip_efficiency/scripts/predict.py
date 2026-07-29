"""EC124 — Liquid Air Energy Storage (LAES / CES) — F1a — Standardized Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import LAESF1a


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = LAESF1a(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        inputs:
            mode             : str — "charge", "discharge", or "idle"
            m_dot_liquid_kgs : float or array, liquid air mass flow [kg/s] (optional)
            soc              : float or array, current SOC [-] (optional, default 0.5)
            time_hours       : float, standby time for boil-off [h] (optional, default 0)
        returns:
            power_kw              : electrical power at terminals [kW] (+charge, -discharge)
            liquid_mass_kg        : liquid air mass in tank [kg]
            tank_volume_m3        : liquid volume [m3]
            soc_after_standby     : SOC after boil-off over time_hours [-]
            energy_capacity_kwh   : usable electrical energy capacity [kWh]
            round_trip_eta        : round-trip electrical efficiency [-]
        """
        mode = inputs.get("mode", "idle").strip().lower()
        m_dot = np.asarray(inputs.get("m_dot_liquid_kgs", 0.0), dtype=float)
        soc = np.asarray(inputs.get("soc", 0.5), dtype=float)
        t_h = float(inputs.get("time_hours", 0.0))

        if mode == "charge":
            power = self._model.charge_power(m_dot)              # +kW into liquefier
        elif mode == "discharge":
            power = -self._model.discharge_power(m_dot)          # -kW (delivered)
        elif mode == "idle":
            power = np.zeros_like(m_dot)
        else:
            raise ValueError(f"mode must be 'charge', 'discharge', or 'idle', got '{mode}'")

        return {
            "power_kw": power,
            "liquid_mass_kg": self._model.liquid_mass(soc),
            "tank_volume_m3": self._model.tank_volume(soc),
            "soc_after_standby": self._model.soc_after_standby(soc, t_h),
            "energy_capacity_kwh": float(self._model.energy_capacity_kwh()),
            "round_trip_eta": float(self._model.round_trip_efficiency()),
        }

    def get_info(self) -> dict:
        return {
            "name": "Liquid Air Energy Storage (LAES / CES)",
            "ec_id": "EC124",
            "fidelity": "F1a",
            "description": (
                "LAES round-trip model: P_charge = m_dot*w_liq*3600/eta_liq; "
                "P_discharge = m_dot*w_disch*3600*eta_p*eta_e*eta_g; "
                "boil-off self-discharge ~0.5%/day; SOC = m_liquid / m_tank_max"
            ),
            "inputs": {
                "mode": {"unit": "-", "values": ["charge", "discharge", "idle"]},
                "m_dot_liquid_kgs": {"unit": "kg/s", "range": [0.0, 200.0]},
                "soc": {"unit": "-", "range": [0.0, 1.0]},
                "time_hours": {"unit": "h", "range": [0.0, 168.0]},
            },
            "outputs": {
                "power_kw": {"unit": "kW"},
                "liquid_mass_kg": {"unit": "kg"},
                "tank_volume_m3": {"unit": "m3"},
                "soc_after_standby": {"unit": "-"},
                "energy_capacity_kwh": {"unit": "kWh"},
                "round_trip_eta": {"unit": "dimensionless"},
            },
            "source": "Morgan et al. (2015), Applied Energy, 137, 845-853; Sciacovelli et al. (2017), Applied Energy, 190, 84-98",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    r_c = model.predict({"mode": "charge", "m_dot_liquid_kgs": 30.0})
    r_d = model.predict({"mode": "discharge", "m_dot_liquid_kgs": 60.0, "soc": 0.5})
    r_i = model.predict({"mode": "idle", "soc": 1.0, "time_hours": 24.0})
    print(f"Charge P_in    : {float(r_c['power_kw'])/1000:.2f} MW (m_dot=30 kg/s)")
    print(f"Discharge P_out: {-float(r_d['power_kw'])/1000:.2f} MW (m_dot=60 kg/s)")
    print(f"Energy capacity: {r_c['energy_capacity_kwh']/1000:.1f} MWh")
    print(f"Round-trip eta : {r_c['round_trip_eta']*100:.2f} %")
    print(f"After 24 h idle from SOC=1.0: SOC = {float(r_i['soc_after_standby']):.4f}")
