"""
EC123 — Compressed Air Energy Storage (CAES) — F1b Thermal Model

Extends F1a (round-trip efficiency) by adding:

1. Cavern thermal drift to rock wall:
   - Cavern air heats up after compression cycle, cools toward rock temperature
   - Newton cooling: dT_cav/dt = -(UA_cav_rock / Cm_cav) * (T_cav - T_rock)
   - After thermal equilibration, T_cav → T_rock (not T_amb)
   - Thermal equilibration time: tau_cav = Cm_cav / UA_cav_rock

2. Ambient temperature effect on compressor intake:
   - Hot air is less dense → compressor must process more volume per kg
   - Specific compression work correction:
       w_comp(T_amb) = w_comp_ref * (1 + k_comp_T * (T_amb - T_ref))
   - Also affects mass flow capacity (iso-corrected flow)

3. Cavern temperature after charging cycle:
   - Compression heat deposited into cavern air
   - T_cav_post_charge = T_cav + Q_heat / Cm_cav
   - Q_heat = m_added * cp_air * (T_comp_out - T_cav)

4. SOC and pressure remain coupled to cavern temperature:
   - At a given pressure, T_cav determines actual air mass:
       m(P, T) = P * V / (R_air * T_cav)

References:
    Budt, M. et al. (2016). A review on compressed air energy storage. Applied Energy, 170, 250-268.
    Greenblatt, J.B. et al. (2012). J. Power Sources, 216, 105-115.
    Succar, S. & Williams, R.H. (2008). Compressed Air Energy Storage.
    Princeton Environmental Institute report.
"""

import numpy as np


class CAESF1b:
    """CAES — thermal model with cavern heat loss to rock and T_amb compressor intake effect."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.eta_comp = u["eta_compressor"]["value"]
        self.eta_motor = u["eta_motor"]["value"]
        self.eta_exp = u["eta_expander"]["value"]
        self.eta_gen = u["eta_generator"]["value"]
        self.heat_rate = u["heat_rate"]["value"]        # kJ/kWh_e
        self.fuel_lhv = u["fuel_lhv"]["value"]          # kJ/kg
        self.p_max = u["p_max"]["value"]                # Pa
        self.p_min = u["p_min"]["value"]                # Pa
        self.V = u["cavern_volume"]["value"]             # m3
        self.T_cav_nominal = u["T_cavern_nominal"]["value"]   # K
        self.T_rock = u["T_rock"]["value"]              # K
        self.w_comp_ref = u["specific_work"]["value"]   # kJ/kg
        self.w_exp = u["specific_expansion"]["value"]   # kJ/kg
        self.R = u["R_air"]["value"]                    # J/(kg·K)
        self.Cm_cav = u["cavern_thermal_mass"]["value"] # J/K
        self.UA_cav = u["UA_cavern_rock"]["value"]      # W/K
        self.T_ref_comp = u["T_ref_comp"]["value"]      # K
        self.k_comp_T = u["k_comp_T"]["value"]          # 1/K
        self.cp_air = u["cp_air"]["value"]              # J/(kg·K)

        # Derived
        self.tau_cav = self.Cm_cav / self.UA_cav        # s
        self.m_max = self.p_max * self.V / (self.R * self.T_cav_nominal)
        self.m_min = self.p_min * self.V / (self.R * self.T_cav_nominal)
        self.m_usable = self.m_max - self.m_min

    # ------------------------------------------------------------------
    # Cavern thermal state
    # ------------------------------------------------------------------

    def cavern_temperature_drift(self, T_cav_0, t_s):
        """
        Cavern temperature [K] after idle time t_s, cooling toward rock temperature.

        T_cav(t) = T_rock + (T_cav_0 - T_rock) * exp(-t / tau_cav)

        Args:
            T_cav_0: Initial cavern temperature [K]
            t_s:     Elapsed time [s]
        """
        T0 = np.asarray(T_cav_0, dtype=float)
        t = np.asarray(t_s, dtype=float)
        return self.T_rock + (T0 - self.T_rock) * np.exp(-t / self.tau_cav)

    def thermal_equilibration_time(self):
        """Time constant for cavern-to-rock heat exchange [s]."""
        return self.tau_cav

    # ------------------------------------------------------------------
    # Air mass and SOC with T_cav
    # ------------------------------------------------------------------

    def air_mass(self, soc, T_cav=None):
        """
        Air mass [kg] in cavern at given SOC and cavern temperature.

        Args:
            soc:    State of charge [0-1]
            T_cav:  Cavern temperature [K] (default: T_cav_nominal)
        """
        T = self.T_cav_nominal if T_cav is None else np.asarray(T_cav, dtype=float)
        s = np.clip(np.asarray(soc, dtype=float), 0.0, 1.0)
        # Recalculate m_max and m_min at this T_cav
        m_max_T = self.p_max * self.V / (self.R * T)
        m_min_T = self.p_min * self.V / (self.R * T)
        return m_min_T + s * (m_max_T - m_min_T)

    def cavern_pressure(self, soc, T_cav=None):
        """Cavern pressure [Pa] at given SOC and T_cav."""
        m = self.air_mass(soc, T_cav)
        T = self.T_cav_nominal if T_cav is None else np.asarray(T_cav, dtype=float)
        return m * self.R * T / self.V

    def soc_from_pressure(self, pressure_Pa, T_cav=None):
        """SOC from measured cavern pressure and temperature."""
        T = self.T_cav_nominal if T_cav is None else float(T_cav)
        m_max_T = self.p_max * self.V / (self.R * T)
        m_min_T = self.p_min * self.V / (self.R * T)
        m = pressure_Pa * self.V / (self.R * T)
        return np.clip((m - m_min_T) / (m_max_T - m_min_T), 0.0, 1.0)

    # ------------------------------------------------------------------
    # Compressor intake temperature correction
    # ------------------------------------------------------------------

    def specific_compression_work(self, T_amb_K=None):
        """
        Specific compression work [kJ/kg] corrected for ambient temperature.

        Hotter intake → less dense air → more work per kg compressed:
            w_comp(T_amb) = w_comp_ref * (1 + k_comp_T * (T_amb - T_ref))

        Args:
            T_amb_K: Ambient temperature [K] (default: T_ref_comp)
        """
        if T_amb_K is None:
            return self.w_comp_ref
        T = np.asarray(T_amb_K, dtype=float)
        return self.w_comp_ref * (1.0 + self.k_comp_T * (T - self.T_ref_comp))

    def charge_power(self, m_dot_air, T_amb_K=None):
        """
        Electrical input power [kW] for compression.

        Includes T_amb correction on specific work.

        Args:
            m_dot_air: Air mass flow rate [kg/s]
            T_amb_K:   Ambient temperature [K]
        """
        m_dot = np.asarray(m_dot_air, dtype=float)
        w = self.specific_compression_work(T_amb_K)
        return m_dot * w / (self.eta_comp * self.eta_motor)

    def discharge_power(self, m_dot_air):
        """Electrical output power [kW] from expansion."""
        m_dot = np.asarray(m_dot_air, dtype=float)
        return m_dot * self.w_exp * self.eta_exp * self.eta_gen

    # ------------------------------------------------------------------
    # Round-trip efficiency with T_amb correction
    # ------------------------------------------------------------------

    def round_trip_efficiency(self, T_amb_K=None):
        """
        Diabatic RTE including T_amb compressor penalty:
            eta_RT = E_out / (E_in_elec + E_in_fuel)
        """
        w_c = self.specific_compression_work(T_amb_K)
        E_out = self.w_exp * self.eta_exp * self.eta_gen / 3600.0
        E_in_elec = w_c / (self.eta_comp * self.eta_motor) / 3600.0
        E_in_fuel = E_out * self.heat_rate / 3600.0
        return E_out / (E_in_elec + E_in_fuel)

    def electric_rte(self, T_amb_K=None):
        """Electricity-only RTE (can exceed 1 for diabatic CAES)."""
        w_c = self.specific_compression_work(T_amb_K)
        E_out = self.w_exp * self.eta_exp * self.eta_gen
        E_in_elec = w_c / (self.eta_comp * self.eta_motor)
        return E_out / E_in_elec

    # ------------------------------------------------------------------
    # Cavern temperature after charge cycle
    # ------------------------------------------------------------------

    def cavern_temp_post_charge(self, soc_before, soc_after, T_cav_before, T_amb_K=None):
        """
        Cavern temperature after adding air from soc_before to soc_after.

        Compression heat deposited = m_added * cp_air * (T_comp_out - T_cav_before)
        T_comp_out ≈ T_amb + (w_comp * eta_comp) / cp_air   (approximate)
        New T_cav = T_cav_before + Q_heat / Cm_cav

        Args:
            soc_before:   SOC before charging
            soc_after:    SOC after charging
            T_cav_before: Cavern T before charging [K]
            T_amb_K:      Ambient temperature [K]
        Returns:
            T_cav_after [K]
        """
        if T_amb_K is None:
            T_amb_K = self.T_ref_comp
        T_amb = float(T_amb_K)
        T_cav0 = float(T_cav_before)

        m_before = self.air_mass(soc_before, T_cav0)
        m_after = self.air_mass(soc_after, T_cav0)
        dm = max(m_after - m_before, 0.0)

        # Temperature of compressed air entering cavern
        w_comp = float(self.specific_compression_work(T_amb))  # kJ/kg
        # Approximate outlet temperature from isentropic compression
        T_comp_out = T_amb + w_comp * 1000.0 * self.eta_comp / self.cp_air

        Q_heat = dm * self.cp_air * max(T_comp_out - T_cav0, 0.0)  # J
        dT = Q_heat / self.Cm_cav
        return T_cav0 + dT

    # ------------------------------------------------------------------
    # Energy capacity
    # ------------------------------------------------------------------

    def energy_capacity_kwh(self, T_cav=None):
        """Usable electrical energy capacity [kWh] at given cavern temperature."""
        T = self.T_cav_nominal if T_cav is None else float(T_cav)
        m_max_T = self.p_max * self.V / (self.R * T)
        m_min_T = self.p_min * self.V / (self.R * T)
        m_usable_T = m_max_T - m_min_T
        return m_usable_T * self.w_exp * self.eta_exp * self.eta_gen / 3600.0
