"""EC125 — Adiabatic CAES (A-CAES) — F1a Round-Trip Model — Standardized Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import ACAESF1a


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = ACAESF1a(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        inputs:
            mode          : str — "charge", "discharge", or "idle"
            m_dot_air     : float or array, air mass flow rate [kg/s] (optional)
            soc           : float or array, current state of charge [-] (optional, default 0.5)
        returns:
            power_kw              : electrical power at terminals [kW] (+charge, -discharge)
            fuel_power_kw         : supplemental fuel thermal input [kW] — always 0 for A-CAES
            fuel_mass_flow_kgs    : fuel mass flow [kg/s] — always 0 for A-CAES
            tes_heat_kw           : heat stored in TES during charge [kW_th]
            cavern_pressure_pa    : cavern pressure at given SOC [Pa]
            cavern_air_mass_kg    : air mass in cavern [kg]
            energy_capacity_kwh   : usable electrical energy capacity [kWh]
            round_trip_eta        : electricity-only round-trip efficiency [-] (~0.65-0.72)
            electric_rt_ratio     : alias for round_trip_eta (same value, no fuel)
        """
        mode  = inputs.get("mode", "idle").strip().lower()
        m_dot = np.asarray(inputs.get("m_dot_air", 0.0), dtype=float)
        soc   = np.asarray(inputs.get("soc", 0.5),       dtype=float)

        if mode == "charge":
            power     = self._model.charge_power(m_dot)         # +kW into compressor
            fuel_p    = self._model.fuel_power(power)            # zeros
            fuel_m    = self._model.fuel_mass_flow(power)        # zeros
            tes_heat  = self._model.tes_heat_stored_kw(m_dot)
        elif mode == "discharge":
            p_out     = self._model.discharge_power(m_dot)
            power     = -p_out                                   # -kW (delivered to grid)
            fuel_p    = self._model.fuel_power(p_out)            # zeros
            fuel_m    = self._model.fuel_mass_flow(p_out)        # zeros
            tes_heat  = np.zeros_like(p_out)
        elif mode == "idle":
            power     = np.zeros_like(m_dot)
            fuel_p    = np.zeros_like(m_dot)
            fuel_m    = np.zeros_like(m_dot)
            tes_heat  = np.zeros_like(m_dot)
        else:
            raise ValueError(f"mode must be 'charge', 'discharge', or 'idle', got '{mode}'")

        rte = float(self._model.round_trip_efficiency())
        return {
            "power_kw":              power,
            "fuel_power_kw":         fuel_p,
            "fuel_mass_flow_kgs":    fuel_m,
            "tes_heat_kw":           tes_heat,
            "cavern_pressure_pa":    self._model.cavern_pressure(soc),
            "cavern_air_mass_kg":    self._model.air_mass(soc),
            "energy_capacity_kwh":   float(self._model.energy_capacity_kwh()),
            "round_trip_eta":        rte,
            "electric_rt_ratio":     rte,
        }

    def get_info(self) -> dict:
        return {
            "name": "Adiabatic Compressed Air Energy Storage (A-CAES)",
            "ec_id": "EC125",
            "fidelity": "F1a",
            "description": (
                "A-CAES round-trip model: heat of compression stored in TES, returned on discharge — no fuel; "
                "P_charge = m_dot*w_comp/(eta_c*eta_m); "
                "P_discharge = m_dot*w_exp_ad*eta_e*eta_g; "
                "RTE = E_elec_out / E_elec_in (~65-72%); "
                "No supplemental fuel — key distinction from diabatic EC123"
            ),
            "inputs": {
                "mode":      {"unit": "-", "values": ["charge", "discharge", "idle"]},
                "m_dot_air": {"unit": "kg/s", "range": [0.0, 500.0]},
                "soc":       {"unit": "-", "range": [0.0, 1.0]},
            },
            "outputs": {
                "power_kw":            {"unit": "kW"},
                "fuel_power_kw":       {"unit": "kW (always 0 — no fuel)"},
                "fuel_mass_flow_kgs":  {"unit": "kg/s (always 0)"},
                "tes_heat_kw":         {"unit": "kW_th (heat to/from TES)"},
                "cavern_pressure_pa":  {"unit": "Pa"},
                "cavern_air_mass_kg":  {"unit": "kg"},
                "energy_capacity_kwh": {"unit": "kWh"},
                "round_trip_eta":      {"unit": "dimensionless (no fuel)"},
                "electric_rt_ratio":   {"unit": "dimensionless (same as round_trip_eta)"},
            },
            "source": (
                "Barbour et al. (2015), Renew. Sust. Energy Rev. 45, 598-614; "
                "Budt et al. (2016), Applied Energy 170, 250-268; "
                "Wolf & Budt (2014), Applied Energy 125, 158-164"
            ),
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    r_c = model.predict({"mode": "charge",    "m_dot_air": 100.0, "soc": 0.5})
    r_d = model.predict({"mode": "discharge", "m_dot_air": 400.0, "soc": 0.5})
    print(f"Charge   P_in  : {float(r_c['power_kw'])/1000:.2f} MW")
    print(f"TES heat stored: {float(r_c['tes_heat_kw'])/1000:.2f} MW_th")
    print(f"Discharge P_out: {-float(r_d['power_kw'])/1000:.2f} MW")
    print(f"Fuel power     : {float(r_d['fuel_power_kw']):.1f} kW  (should be 0)")
    print(f"Cavern pressure @SOC=0.5: {float(r_c['cavern_pressure_pa'])/1e5:.2f} bar")
    print(f"Energy capacity: {r_c['energy_capacity_kwh']/1000:.2f} MWh")
    print(f"Round-trip eta : {r_c['round_trip_eta']:.4f}")
