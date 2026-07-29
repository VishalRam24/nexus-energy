"""EC124 — LAES — F1b Thermal — Standardized Predict Interface"""
import json
import numpy as np
from pathlib import Path
from model import LAESF1b


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = LAESF1b(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        inputs:
            soc             : float or array [0-1]
            m_dot_liquid    : liquid air mass flow [kg/s]
            T_amb_K         : ambient temperature [K] (default T_amb_ref)
            time_hours      : standby duration for idle mode [h]
            mode            : 'charge' | 'discharge' | 'idle' (default 'discharge')
        returns (charge/discharge):
            power_kw, round_trip_efficiency, boil_off_rate_per_day,
            cold_recycle_effectiveness, effective_discharge_work_kwh_kg,
            effective_liquefaction_work_kwh_kg, energy_capacity_kwh
        returns (idle):
            soc_after, liquid_mass_kg, boil_off_fraction
        """
        m = self._model
        mode = inputs.get("mode", "discharge")
        soc = np.asarray(inputs.get("soc", 0.5), dtype=float)
        T_amb = inputs.get("T_amb_K", None)
        m_dot = np.asarray(inputs.get("m_dot_liquid", 0.0), dtype=float)

        if mode == "idle":
            t_h = float(inputs.get("time_hours", 24.0))
            soc_after = m.soc_after_standby(soc, t_h, T_amb)
            m_liq = m.liquid_mass(soc_after)
            bo_frac = m.boil_off_loss_fraction(t_h, T_amb)
            return {
                "soc_after": soc_after,
                "liquid_mass_kg": m_liq,
                "boil_off_fraction": bo_frac,
                "boil_off_rate_per_day": m.boil_off_rate_per_day(T_amb),
            }

        if mode == "charge":
            P_kw = m.charge_power(m_dot, T_amb)
        else:
            P_kw = m.discharge_power(m_dot, T_amb)

        return {
            "power_kw": P_kw,
            "round_trip_efficiency": m.round_trip_efficiency(T_amb),
            "boil_off_rate_per_day": m.boil_off_rate_per_day(T_amb),
            "cold_recycle_effectiveness": m.cold_recycle_effectiveness(T_amb),
            "effective_discharge_work_kwh_kg": m.effective_discharge_work(T_amb),
            "effective_liquefaction_work_kwh_kg": m.effective_liquefaction_work(T_amb),
            "energy_capacity_kwh": m.energy_capacity_kwh(T_amb),
        }

    def get_info(self) -> dict:
        u = self.params["unit"]
        return {
            "name": "Liquid Air Energy Storage (LAES) — Thermal",
            "ec_id": "EC124",
            "fidelity": "F1b",
            "description": (
                "T_amb-dependent boil-off rate (BOR ~0.5%/day, increases with T); "
                "cold recycle effectiveness vs ambient T; "
                "discharge specific work vs T_amb; full thermal RTE."
            ),
            "inputs": {
                "soc": {"unit": "dimensionless", "range": [0.0, 1.0]},
                "m_dot_liquid": {"unit": "kg/s", "range": [0.0, 200.0]},
                "T_amb_K": {"unit": "K", "range": [253.15, 323.15]},
                "time_hours": {"unit": "h", "note": "idle mode"},
                "mode": {"values": ["charge", "discharge", "idle"]},
            },
            "outputs": {
                "power_kw": {"unit": "kW"},
                "round_trip_efficiency": {"unit": "dimensionless"},
                "boil_off_rate_per_day": {"unit": "fraction/day"},
                "cold_recycle_effectiveness": {"unit": "dimensionless"},
                "effective_discharge_work_kwh_kg": {"unit": "kWh/kg"},
                "effective_liquefaction_work_kwh_kg": {"unit": "kWh/kg"},
                "energy_capacity_kwh": {"unit": "kWh"},
                "soc_after": {"unit": "dimensionless", "mode": "idle"},
                "boil_off_fraction": {"unit": "dimensionless", "mode": "idle"},
            },
            "params": {
                "bor_ref": f"{u['boil_off_per_day_ref']['value'] * 100:.2f} %/day at T_ref",
                "T_amb_ref": f"{u['T_amb_ref']['value']} K",
                "cold_recycle_eff_ref": u["cold_recycle_eff_ref"]["value"],
            },
            "source": "Morgan et al. (2015); Sciacovelli et al. (2017); Guizzi et al. (2015)",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    print("=== EC124 F1b LAES Thermal Model ===\n")
    print("BOR and cold recycle vs T_amb:")
    for T in [253.15, 273.15, 298.15, 313.15, 323.15]:
        r = model.predict({"soc": 0.8, "m_dot_liquid": 50.0, "T_amb_K": T, "mode": "discharge"})
        print(f"  {T - 273.15:+.0f}°C: BOR={float(r['boil_off_rate_per_day']) * 100:.3f}%/day  "
              f"eps={float(r['cold_recycle_effectiveness']):.3f}  "
              f"RTE={float(r['round_trip_efficiency']):.4f}")

    print("\nBoil-off after standby (soc=0.8, T_amb=298 K):")
    for t_h in [24, 48, 168, 336]:
        r = model.predict({"soc": 0.8, "T_amb_K": 298.15, "time_hours": t_h, "mode": "idle"})
        print(f"  t={t_h:>4d} h: SOC={float(r['soc_after']):.4f}  "
              f"loss={float(r['boil_off_fraction']) * 100:.2f}%")
