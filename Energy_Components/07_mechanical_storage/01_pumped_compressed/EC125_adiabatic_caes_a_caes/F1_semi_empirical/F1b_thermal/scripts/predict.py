"""EC125 — A-CAES — F1b Thermal — Standardized Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import ACAESF1b


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = ACAESF1b(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        inputs:
            mode          : 'charge' | 'discharge' | 'idle' | 'thermal'
            m_dot_air     : air mass flow [kg/s] (required for power modes)
            soc           : float or array [0-1]
            T_amb_K       : ambient temperature [K] (default: T_ref=288.15)
            T_cav_K       : cavern temperature [K]  (default: T_cav_nominal)
            T_tes_K       : TES temperature [K]     (default: T_tes_design)
            T_cav_0_K     : initial cavern T [K]    (mode='thermal')
            T_tes_0_K     : initial TES T [K]       (mode='thermal')
            t_idle_s      : idle time [s]           (mode='thermal')
        returns:
            power_kw, fuel_power_kw, tes_heat_kw, cavern_pressure_pa,
            cavern_air_mass_kg, energy_capacity_kwh, round_trip_eta,
            specific_comp_work_kj_kg, expansion_work_eff_kj_kg
        """
        m = self._model
        mode = inputs.get("mode", "discharge")
        soc = np.asarray(inputs.get("soc", 0.5), dtype=float)
        T_amb = inputs.get("T_amb_K", m.T_ref_comp)
        T_cav = inputs.get("T_cav_K", m.T_cav_nom)
        T_tes = inputs.get("T_tes_K", None)   # None → design temperature
        m_dot = np.asarray(inputs.get("m_dot_air", 0.0), dtype=float)

        if mode == "thermal":
            t_idle = float(inputs.get("t_idle_s", 3600.0))
            T_cav_0 = float(inputs.get("T_cav_0_K", m.T_cav_nom))
            T_tes_0 = float(inputs.get("T_tes_0_K", m.T_tes_design))
            T_cav_new = m.cavern_temperature_drift(T_cav_0, t_idle)
            T_tes_new = m.tes_temperature_after_idle(T_tes_0, t_idle)
            return {
                "T_cav_K": float(T_cav_new),
                "T_tes_K": float(T_tes_new),
                "tau_cav_s": m.tau_cav,
                "tau_tes_s": m.tau_tes,
                "tes_heat_fraction": float(m.tes_heat_available_fraction(T_tes_new)),
            }

        if mode == "charge":
            P_kw = m.charge_power(m_dot, T_amb)
            tes_heat = m.tes_heat_stored_kw(m_dot, T_amb)
            fuel_kw = 0.0
        elif mode == "discharge":
            P_kw = m.discharge_power(m_dot, T_tes)
            tes_heat = 0.0
            fuel_kw = 0.0
        elif mode == "idle":
            P_kw = np.zeros_like(m_dot)
            tes_heat = 0.0
            fuel_kw = 0.0
        else:
            raise ValueError(f"mode must be 'charge','discharge','idle','thermal'; got '{mode}'")

        return {
            "power_kw": P_kw,
            "fuel_power_kw": np.asarray(fuel_kw, dtype=float),
            "fuel_mass_flow_kgs": np.zeros_like(m_dot),
            "tes_heat_kw": np.asarray(tes_heat, dtype=float),
            "cavern_pressure_pa": m.cavern_pressure(soc, T_cav),
            "cavern_air_mass_kg": m.air_mass(soc, T_cav),
            "energy_capacity_kwh": m.energy_capacity_kwh(T_cav, T_tes),
            "round_trip_eta": m.round_trip_efficiency(T_amb, T_tes),
            "specific_comp_work_kj_kg": m.specific_compression_work(T_amb),
            "expansion_work_eff_kj_kg": m.expansion_work_effective(T_tes),
        }

    def get_info(self) -> dict:
        u = self.params["unit"]
        return {
            "name": "Adiabatic CAES (A-CAES) — Thermal",
            "ec_id": "EC125",
            "fidelity": "F1b",
            "description": (
                "Extends F1a with TES thermal decay, T_amb compressor correction, "
                "and cavern thermal drift. RTE calibrated to 0.700 at design; "
                "physical limit < 0.75 enforced."
            ),
            "inputs": {
                "soc": {"unit": "dimensionless", "range": [0.0, 1.0]},
                "m_dot_air": {"unit": "kg/s", "range": [0.0, 500.0]},
                "T_amb_K": {"unit": "K", "range": [253.15, 313.15]},
                "T_cav_K": {"unit": "K", "range": [288.15, 333.15]},
                "T_tes_K": {"unit": "K", "range": [298.15, 600.0]},
                "mode": {"values": ["charge", "discharge", "idle", "thermal"]},
            },
            "outputs": {
                "power_kw": {"unit": "kW"},
                "tes_heat_kw": {"unit": "kW_th"},
                "cavern_pressure_pa": {"unit": "Pa"},
                "cavern_air_mass_kg": {"unit": "kg"},
                "energy_capacity_kwh": {"unit": "kWh"},
                "round_trip_eta": {"unit": "dimensionless"},
                "expansion_work_eff_kj_kg": {"unit": "kJ/kg"},
            },
            "source": u["source"],
            "rte_note": "A-CAES design RTE=0.700; physical limit <0.75 (no fuel)",
        }


if __name__ == "__main__":
    model = ComponentModel()
    print("=== EC125 A-CAES F1b Thermal ===\n")
    print("RTE vs T_amb:")
    for T in [253.15, 273.15, 288.15, 303.15, 313.15]:
        r = model.predict({"mode": "discharge", "m_dot_air": 400.0, "T_amb_K": T, "soc": 0.5})
        print(f"  T_amb={T-273.15:+.0f}°C  RTE={float(r['round_trip_eta']):.4f}")

    print("\nTES thermal decay:")
    for t_h in [0, 6, 24, 56, 168]:
        r = model.predict({"mode": "thermal", "T_tes_0_K": 550.0, "t_idle_s": t_h * 3600})
        print(f"  t={t_h:>4d}h  T_tes={float(r['T_tes_K']):.1f} K  "
              f"f_heat={float(r['tes_heat_fraction']):.3f}")
