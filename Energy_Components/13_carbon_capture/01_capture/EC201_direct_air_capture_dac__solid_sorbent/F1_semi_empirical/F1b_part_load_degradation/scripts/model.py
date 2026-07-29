"""
EC201 — Direct Air Capture (DAC) Solid Sorbent — F1b Part-Load + Degradation Model

Extends F1a energy model with:
  1. Sorbent degradation: q(n) = q0 * (1 - k_deg * n_cycles), k_deg ~ 5e-5/cycle.
     After 20,000 cycles (~2.3 years at 1h cycles): ~0% capacity remaining.
  2. Humidity effect on CO2 capacity: q_factor = a + b*RH + c*RH^2.
     Low humidity reduces amine sorbent performance.
  3. Temperature swing energy: E_th = m_sorbent * cp * (T_desorption - T_ambient) / CO2_captured.
  4. Part-load: reduced air flow → proportionally less capture, fan energy penalty.

Reference:
    Fasihi et al. (2019). J. Cleaner Production, 224, 957-980.
    Sinha, A. et al. (2017). Ind. Eng. Chem. Res., 56(3), 750-764.
"""

import numpy as np


class DACF1b:
    """Solid-sorbent DAC — part-load + sorbent degradation model."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.q0 = u["q0"]["value"]  # mmol_CO2/g_sorbent
        self.sorbent_mass_kg = u["sorbent_mass_kg"]["value"]
        self.T_desorption = u["T_desorption_degC"]["value"]
        self.cycle_time_s = u["cycle_time_s"]["value"]
        self.k_deg = u["k_deg"]["value"]
        self.E_th_base = u["E_th_base"]["value"]
        self.E_el_base = u["E_el_base"]["value"]
        self.capture_efficiency = u["capture_efficiency"]["value"]
        self.CO2_ppm = u["CO2_concentration_ppm"]["value"]
        self.rho_air = u["rho_air"]["value"]
        self.M_CO2 = u["CO2_molar_mass"]["value"]
        self.M_air = u["air_molar_mass"]["value"]
        self.cp_sorbent = u["cp_sorbent"]["value"]
        self.humidity_coeffs = u["humidity_capacity_coeffs"]["value"]
        # Mass concentration of CO2 in air [kg_CO2/m3_air]
        self.CO2_conc_kgm3 = (self.CO2_ppm * 1e-6) * (self.M_CO2 / self.M_air) * self.rho_air

    def sorbent_capacity_pct(self, n_cycles):
        """Remaining sorbent capacity (%) after n_cycles.
        q(n) = q0 * (1 - k_deg * n)
        """
        n = np.asarray(n_cycles, dtype=float)
        frac = np.clip(1.0 - self.k_deg * n, 0.0, 1.0)
        return frac * 100.0

    def _capacity_factor(self, n_cycles):
        """Remaining capacity as fraction [0, 1]."""
        n = np.asarray(n_cycles, dtype=float)
        return np.clip(1.0 - self.k_deg * n, 0.0, 1.0)

    def _humidity_factor(self, relative_humidity):
        """Humidity effect on sorbent CO2 uptake capacity.
        q_factor = a + b*RH + c*RH^2  (normalized so ~1.0 at RH=0.5)
        """
        rh = np.asarray(relative_humidity, dtype=float)
        a, b, c = self.humidity_coeffs
        f = a + b * rh + c * rh ** 2
        # Normalize so at RH=0.5: factor=1.0
        f_ref = a + b * 0.5 + c * 0.25
        return np.clip(f / f_ref, 0.3, 1.3)

    def co2_captured_kg_h(self, air_flow_m3_s, T_ambient_degC, relative_humidity,
                          plr, n_cycles):
        """CO2 captured (kg/h) accounting for capacity degradation and humidity.

        Method: min of (air-side capture, sorbent-side capacity per cycle).

        Air-side: CO2_rate = air_flow * PLR * CO2_conc * capture_efficiency
        Sorbent-side: CO2_per_cycle = q * humidity_factor * capacity_factor * m_sorbent
        """
        flow = np.asarray(air_flow_m3_s, dtype=float)
        plr = np.asarray(plr, dtype=float)
        rh = np.asarray(relative_humidity, dtype=float)
        n = np.asarray(n_cycles, dtype=float)

        # Air-side CO2 available
        co2_air_kgs = flow * plr * self.CO2_conc_kgm3 * self.capture_efficiency
        co2_air_kgh = co2_air_kgs * 3600.0

        # Sorbent-side capacity per cycle
        cap_f = self._capacity_factor(n)
        hum_f = self._humidity_factor(rh)
        # q0 in mmol/g -> mol/kg: q0 * 1e-3 * 1000 = q0 (same number)
        # CO2 per cycle = q0 * cap_f * hum_f * m_sorbent [mmol/g * g_to_kg] = q0 * cap_f * hum_f * m * 1e3 [mmol]
        # Convert to kg: * M_CO2/1e6
        co2_sorbent_kg_per_cycle = (self.q0 * cap_f * hum_f *
                                    self.sorbent_mass_kg * 1e3 *
                                    self.M_CO2 / 1e6)
        # Cycles per hour
        cycles_per_hour = 3600.0 / self.cycle_time_s
        co2_sorbent_kgh = co2_sorbent_kg_per_cycle * cycles_per_hour

        return np.minimum(co2_air_kgh, co2_sorbent_kgh)

    def thermal_energy_kwh_ton(self, T_ambient_degC, relative_humidity, n_cycles):
        """Thermal energy for temperature swing regeneration (kWh/tCO2).
        E_th = base + m_sorbent * cp * dT / CO2_per_cycle [kJ -> kWh/tCO2]
        """
        T_amb = np.asarray(T_ambient_degC, dtype=float)
        dT = self.T_desorption - T_amb
        cap_f = self._capacity_factor(n_cycles)
        hum_f = self._humidity_factor(relative_humidity)

        # CO2 per cycle (kg)
        co2_per_cycle = (self.q0 * cap_f * hum_f *
                         self.sorbent_mass_kg * 1e3 *
                         self.M_CO2 / 1e6)
        co2_per_cycle = np.clip(co2_per_cycle, 1e-6, None)

        # Sensible heat for sorbent heating (kJ)
        Q_sensible = self.sorbent_mass_kg * self.cp_sorbent * dT
        # Convert to kWh per tonne CO2
        # Q_sensible [kJ] / co2_per_cycle [kg] * 1000 [kg/t] / 3600 [kJ/kWh]
        E_th_swing = Q_sensible / co2_per_cycle * 1000.0 / 3600.0

        # Total = base (desorption enthalpy) + swing energy
        # At degraded state, base energy also increases (less CO2 per cycle)
        E_th_base_adj = self.E_th_base / (cap_f * hum_f + 1e-6)
        E_th = E_th_base_adj + E_th_swing

        return np.clip(E_th, 500.0, 10000.0)

    def electrical_energy_kwh_ton(self, plr, n_cycles):
        """Electrical energy for fans and vacuum (kWh/tCO2).
        Fan energy scales inversely with PLR (fixed infrastructure).
        """
        plr = np.asarray(plr, dtype=float)
        cap_f = self._capacity_factor(n_cycles)
        # Fan penalty at part-load + degradation penalty
        fan_factor = 0.7 + 0.3 / (plr + 1e-6)
        E_el = self.E_el_base * fan_factor / (cap_f + 1e-6)
        return np.clip(E_el, 100.0, 5000.0)

    def compute(self, air_flow_m3_s, T_ambient_degC, relative_humidity, plr, n_cycles):
        """Full computation returning all outputs."""
        co2 = self.co2_captured_kg_h(air_flow_m3_s, T_ambient_degC,
                                     relative_humidity, plr, n_cycles)
        E_th = self.thermal_energy_kwh_ton(T_ambient_degC, relative_humidity, n_cycles)
        E_el = self.electrical_energy_kwh_ton(plr, n_cycles)
        cap_pct = self.sorbent_capacity_pct(n_cycles)

        return {
            "co2_captured_kg_h": co2,
            "thermal_energy_kwh_ton": E_th,
            "electrical_energy_kwh_ton": E_el,
            "sorbent_capacity_pct": cap_pct,
        }
