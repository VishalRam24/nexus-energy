"""EC123 — Compressed Air Energy Storage (CAES) — F1a — Standardized Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import CAESF1a


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = CAESF1a(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        inputs:
            mode          : str — "charge", "discharge", or "idle"
            m_dot_air     : float or array, air mass flow rate [kg/s] (optional)
            soc           : float or array, current state of charge [-] (optional, default 0.5)
        returns:
            power_kw              : electrical power at terminals [kW] (+charge, -discharge)
            fuel_power_kw         : supplemental fuel thermal input [kW] (discharge only)
            fuel_mass_flow_kgs    : fuel mass flow [kg/s] (discharge only)
            cavern_pressure_pa    : cavern pressure at given SOC [Pa]
            cavern_air_mass_kg    : air mass in cavern [kg]
            energy_capacity_kwh   : usable electrical energy capacity [kWh]
            round_trip_eta        : diabatic round-trip efficiency (elec_out / (elec_in + fuel_in)) [-]
            electric_rt_ratio     : elec-only ratio (can exceed 1 due to fuel) [-]
        """
        mode = inputs.get("mode", "idle").strip().lower()
        m_dot = np.asarray(inputs.get("m_dot_air", 0.0), dtype=float)
        soc = np.asarray(inputs.get("soc", 0.5), dtype=float)

        if mode == "charge":
            power = self._model.charge_power(m_dot)            # +kW into compressor
            fuel_p = np.zeros_like(power)
            fuel_m = np.zeros_like(power)
        elif mode == "discharge":
            p_out = self._model.discharge_power(m_dot)
            power = -p_out                                     # -kW (delivered to grid)
            fuel_p = self._model.fuel_power(p_out)
            fuel_m = self._model.fuel_mass_flow(p_out)
        elif mode == "idle":
            power = np.zeros_like(m_dot)
            fuel_p = np.zeros_like(m_dot)
            fuel_m = np.zeros_like(m_dot)
        else:
            raise ValueError(f"mode must be 'charge', 'discharge', or 'idle', got '{mode}'")

        return {
            "power_kw": power,
            "fuel_power_kw": fuel_p,
            "fuel_mass_flow_kgs": fuel_m,
            "cavern_pressure_pa": self._model.cavern_pressure(soc),
            "cavern_air_mass_kg": self._model.air_mass(soc),
            "energy_capacity_kwh": float(self._model.energy_capacity_kwh()),
            "round_trip_eta": float(self._model.round_trip_efficiency()),
            "electric_rt_ratio": float(self._model.electric_round_trip_efficiency()),
        }

    def get_info(self) -> dict:
        return {
            "name": "Compressed Air Energy Storage (CAES)",
            "ec_id": "EC123",
            "fidelity": "F1a",
            "description": (
                "Diabatic CAES round-trip model: cavern SOC from p,V,T; "
                "P_charge = m_dot*w_comp/(eta_c*eta_m); "
                "P_discharge = m_dot*w_exp*eta_e*eta_g; "
                "fuel input = P_out*heat_rate; "
                "RTE = E_elec_out / (E_elec_in + E_fuel_in)"
            ),
            "inputs": {
                "mode": {"unit": "-", "values": ["charge", "discharge", "idle"]},
                "m_dot_air": {"unit": "kg/s", "range": [0.0, 500.0]},
                "soc": {"unit": "-", "range": [0.0, 1.0]},
            },
            "outputs": {
                "power_kw": {"unit": "kW"},
                "fuel_power_kw": {"unit": "kW (thermal)"},
                "fuel_mass_flow_kgs": {"unit": "kg/s"},
                "cavern_pressure_pa": {"unit": "Pa"},
                "cavern_air_mass_kg": {"unit": "kg"},
                "energy_capacity_kwh": {"unit": "kWh"},
                "round_trip_eta": {"unit": "dimensionless"},
                "electric_rt_ratio": {"unit": "dimensionless"},
            },
            "source": "Budt et al. (2016), Applied Energy, 170, 250-268; Luo et al. (2015), Applied Energy, 137, 511-536",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    info = model.get_info()
    r_c = model.predict({"mode": "charge", "m_dot_air": 100.0, "soc": 0.5})
    r_d = model.predict({"mode": "discharge", "m_dot_air": 400.0, "soc": 0.5})
    print(f"Charge  P_in  : {float(r_c['power_kw'])/1000:.2f} MW (m_dot=100 kg/s)")
    print(f"Discharge P_out: {-float(r_d['power_kw'])/1000:.2f} MW (m_dot=400 kg/s)")
    print(f"Fuel power     : {float(r_d['fuel_power_kw'])/1000:.2f} MW_th")
    print(f"Cavern pressure @SOC=0.5: {float(r_c['cavern_pressure_pa'])/1e5:.2f} bar")
    print(f"Energy capacity: {r_c['energy_capacity_kwh']/1000:.2f} MWh")
    print(f"Round-trip eta : {r_c['round_trip_eta']:.4f}")
    print(f"Elec-only ratio: {r_c['electric_rt_ratio']:.4f}")
