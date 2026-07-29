"""EC123 — CAES — F1b Thermal — Standardized Predict Interface"""
import json
import numpy as np
from pathlib import Path
from model import CAESF1b


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = CAESF1b(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        inputs:
            soc           : float or array [0-1]
            m_dot_air     : air mass flow rate [kg/s] (required for power)
            T_amb_K       : ambient temperature [K]  (default: T_ref_comp = 288.15 K)
            T_cav_K       : cavern temperature [K]   (default: T_cav_nominal)
            mode          : 'charge' | 'discharge' | 'thermal' (default 'discharge')
            t_idle_s      : idle time for thermal drift [s] (mode='thermal')
        returns:
            power_kw, cavern_pressure_Pa, air_mass_kg, round_trip_efficiency,
            electric_rte, specific_work_kJ_kg, T_cav_drifted (mode='thermal')
        """
        m = self._model
        mode = inputs.get("mode", "discharge")
        soc = np.asarray(inputs.get("soc", 0.5), dtype=float)
        T_amb = inputs.get("T_amb_K", m.T_ref_comp)
        T_cav = inputs.get("T_cav_K", m.T_cav_nominal)
        m_dot = np.asarray(inputs.get("m_dot_air", 0.0), dtype=float)

        if mode == "thermal":
            t_idle = float(inputs.get("t_idle_s", 3600.0))
            T_cav_0 = float(inputs.get("T_cav_0_K", m.T_cav_nominal))
            T_cav_drift = m.cavern_temperature_drift(T_cav_0, t_idle)
            tau = m.thermal_equilibration_time()
            p = m.cavern_pressure(soc, T_cav_drift)
            air = m.air_mass(soc, T_cav_drift)
            return {
                "T_cav_K": float(T_cav_drift),
                "tau_cav_s": tau,
                "cavern_pressure_Pa": float(p),
                "air_mass_kg": float(air),
            }

        if mode == "charge":
            P_kw = m.charge_power(m_dot, T_amb)
        else:
            P_kw = m.discharge_power(m_dot)

        return {
            "power_kw": P_kw,
            "cavern_pressure_Pa": m.cavern_pressure(soc, T_cav),
            "air_mass_kg": m.air_mass(soc, T_cav),
            "round_trip_efficiency": m.round_trip_efficiency(T_amb),
            "electric_rte": m.electric_rte(T_amb),
            "specific_work_kJ_kg": m.specific_compression_work(T_amb),
            "energy_capacity_kwh": m.energy_capacity_kwh(T_cav),
        }

    def get_info(self) -> dict:
        u = self.params["unit"]
        return {
            "name": "Compressed Air Energy Storage (CAES) — Thermal",
            "ec_id": "EC123",
            "fidelity": "F1b",
            "description": (
                "Adds cavern thermal drift to rock wall (Newton cooling), "
                "T_amb effect on compressor intake specific work, "
                "and cavern temperature after charge cycle."
            ),
            "inputs": {
                "soc": {"unit": "dimensionless", "range": [0.0, 1.0]},
                "m_dot_air": {"unit": "kg/s", "range": [0.0, 500.0]},
                "T_amb_K": {"unit": "K", "range": [253.15, 313.15]},
                "T_cav_K": {"unit": "K", "range": [288.15, 333.15]},
                "mode": {"values": ["charge", "discharge", "thermal"]},
            },
            "outputs": {
                "power_kw": {"unit": "kW"},
                "cavern_pressure_Pa": {"unit": "Pa"},
                "air_mass_kg": {"unit": "kg"},
                "round_trip_efficiency": {"unit": "dimensionless"},
                "electric_rte": {"unit": "dimensionless"},
                "specific_work_kJ_kg": {"unit": "kJ/kg"},
                "T_cav_K": {"unit": "K", "mode": "thermal"},
                "tau_cav_s": {"unit": "s", "mode": "thermal"},
            },
            "params": {
                "T_rock": f"{u['T_rock']['value']} K",
                "UA_cavern_rock": f"{u['UA_cavern_rock']['value']} W/K",
                "tau_cav_s": f"{self._model.tau_cav:.0f} s",
                "T_ref_comp": f"{u['T_ref_comp']['value']} K",
                "k_comp_T": f"{u['k_comp_T']['value']} /K",
            },
            "source": "Budt et al. (2016); Greenblatt et al. (2012); Succar & Williams (2008)",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    print("=== EC123 F1b CAES Thermal Model ===\n")
    print("Compressor specific work vs T_amb:")
    for T in [253.15, 273.15, 288.15, 303.15, 313.15]:
        r = model.predict({"soc": 0.5, "m_dot_air": 100.0, "T_amb_K": T, "mode": "charge"})
        print(f"  T_amb={T - 273.15:+.0f}°C  w_comp={float(r['specific_work_kJ_kg']):.1f} kJ/kg  "
              f"RTE={float(r['round_trip_efficiency']):.4f}")

    print("\nCavern thermal drift:")
    for t_h in [0, 1, 6, 24, 48, 168]:
        r = model.predict({"mode": "thermal", "soc": 0.5,
                           "T_cav_0_K": 330.0, "t_idle_s": t_h * 3600})
        print(f"  t={t_h:>4d} h  T_cav={float(r['T_cav_K']):.2f} K")
