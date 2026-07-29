"""
EC125 — Adiabatic Compressed Air Energy Storage (A-CAES) — F1b Thermal Model

Extends F1a round-trip model with:

1. TES thermal losses over time (Newton cooling of the TES store):
   T_tes(t) = T_tes_0 + (T_tes_charge - T_tes_0) * exp(-t / tau_tes)
   - tau_tes = Cm_tes / UA_tes  (TES thermal time constant)
   - Heat returned on discharge is fraction of stored heat (T-dependent)

2. Ambient temperature effect on compressor specific work:
   w_comp(T_amb) = w_comp_ref * (1 + k_comp_T * (T_amb - T_ref))
   - Hotter intake -> more work per kg (less dense air)

3. Cavern air temperature after charge (same as EC123):
   T_cav(t) = T_rock + (T_post - T_rock) * exp(-t / tau_cav)

4. TES-recovered heat effect on expansion work:
   w_exp_effective = w_exp_ref * (T_tes_discharge / T_tes_design)^(gamma/(gamma-1) - 1)
   Simplified linear approximation:
   w_exp_eff = w_exp_ref * (1 - k_tes_T * (T_tes_design - T_tes_actual) / T_tes_design)

5. RTE including all thermal penalties:
   eta_RT = (w_exp_eff * eta_exp * eta_gen) / (w_comp(T_amb) / (eta_comp * eta_motor))

   A-CAES RTE calibrated range: 0.60-0.75 (physical upper bound ~0.75)
   Recalibrated w_exp = 335 kJ/kg gives RTE = 0.700 at design conditions (Phase 7 fix).
   RTE < 0.75 is the hard physical constraint for A-CAES.

References:
    Barbour, E., et al. (2015). Renewable and Sustainable Energy Reviews, 45, 598-614.
    Budt, M., Wolf, D., Span, R., Yan, J. (2016). Applied Energy, 170, 250-268.
    Wolf, D. & Budt, M. (2014). LTA-CAES. Applied Energy, 125, 158-164.
    Zunft, S. et al. (2006). A-CAES and TES modelling, DLR TR.
"""

import numpy as np

_GAMMA = 1.4    # specific heat ratio for air


class ACAESF1b:
    """Adiabatic CAES — thermal model: TES losses + T_amb compressor correction."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.eta_comp   = u["eta_compressor"]["value"]
        self.eta_motor  = u["eta_motor"]["value"]
        self.eta_exp    = u["eta_expander"]["value"]
        self.eta_gen    = u["eta_generator"]["value"]
        self.eta_tes    = u["eta_tes"]["value"]
        self.p_max      = u["p_max"]["value"]          # Pa
        self.p_min      = u["p_min"]["value"]          # Pa
        self.V          = u["cavern_volume"]["value"]   # m3
        self.T_cav_nom  = u["T_cavern_nominal"]["value"]  # K
        self.T_rock     = u["T_rock"]["value"]         # K
        self.R          = u["R_air"]["value"]          # J/(kg·K)

        # Thermal parameters
        self.w_comp_ref = u["specific_work"]["value"]           # kJ/kg
        self.w_exp_ref  = u["specific_expansion_adiabatic"]["value"]  # kJ/kg
        self.T_ref_comp = u["T_ref_comp"]["value"]              # K (ISO 15°C)
        self.k_comp_T   = u["k_comp_T"]["value"]               # 1/K
        self.Cm_cav     = u["cavern_thermal_mass"]["value"]     # J/K
        self.UA_cav     = u["UA_cavern_rock"]["value"]          # W/K
        self.cp_air     = u["cp_air"]["value"]                  # J/(kg·K)

        # TES thermal model
        self.T_tes_design = u["T_tes_design"]["value"]          # K (design charge temp)
        self.T_tes_ambient = u["T_tes_ambient"]["value"]        # K (ambient/idle temp)
        self.Cm_tes       = u["Cm_tes"]["value"]               # J/K
        self.UA_tes       = u["UA_tes"]["value"]               # W/K
        self.k_tes_T      = u["k_tes_temp_coeff"]["value"]     # 1/dimensionless

        # Derived
        self.tau_cav = self.Cm_cav / self.UA_cav    # s
        self.tau_tes = self.Cm_tes / self.UA_tes     # s
        self._m_max  = self.p_max * self.V / (self.R * self.T_cav_nom)
        self._m_min  = self.p_min * self.V / (self.R * self.T_cav_nom)
        self._m_usable = self._m_max - self._m_min

    # ------------------------------------------------------------------
    # TES thermal state
    # ------------------------------------------------------------------

    def tes_temperature_after_idle(self, T_tes_0, t_idle_s):
        """
        TES temperature [K] after idle period t_idle_s (heat loss to environment).

        T_tes(t) = T_tes_ambient + (T_tes_0 - T_tes_ambient) * exp(-t / tau_tes)
        """
        T0 = np.asarray(T_tes_0, dtype=float)
        t = np.asarray(t_idle_s, dtype=float)
        return self.T_tes_ambient + (T0 - self.T_tes_ambient) * np.exp(-t / self.tau_tes)

    def tes_heat_available_fraction(self, T_tes_K):
        """
        Fraction of design expansion work available given TES temperature T_tes_K.

        Linear approximation:
            f = 1 - k_tes_T * (T_tes_design - T_tes) / T_tes_design
        Clamped to [0, 1].
        """
        T = np.asarray(T_tes_K, dtype=float)
        frac = 1.0 - self.k_tes_T * (self.T_tes_design - T) / self.T_tes_design
        return np.clip(frac, 0.0, 1.0)

    def expansion_work_effective(self, T_tes_K=None):
        """
        Effective expansion specific work [kJ/kg] accounting for TES temperature.

        If T_tes_K is None, assumes fully charged TES (design conditions).
        """
        if T_tes_K is None:
            return self.w_exp_ref
        f = self.tes_heat_available_fraction(T_tes_K)
        return self.w_exp_ref * f

    # ------------------------------------------------------------------
    # Cavern thermal state
    # ------------------------------------------------------------------

    def cavern_temperature_drift(self, T_cav_0, t_s):
        """
        Cavern temperature [K] after idle time t_s.

        T_cav(t) = T_rock + (T_cav_0 - T_rock) * exp(-t / tau_cav)
        """
        T0 = np.asarray(T_cav_0, dtype=float)
        t = np.asarray(t_s, dtype=float)
        return self.T_rock + (T0 - self.T_rock) * np.exp(-t / self.tau_cav)

    # ------------------------------------------------------------------
    # Compressor work with T_amb correction
    # ------------------------------------------------------------------

    def specific_compression_work(self, T_amb_K=None):
        """
        Specific compression work [kJ/kg] corrected for ambient temperature.

        w_comp(T_amb) = w_comp_ref * (1 + k_comp_T * (T_amb - T_ref))
        """
        if T_amb_K is None:
            return self.w_comp_ref
        T = np.asarray(T_amb_K, dtype=float)
        return self.w_comp_ref * (1.0 + self.k_comp_T * (T - self.T_ref_comp))

    # ------------------------------------------------------------------
    # Air mass and SOC
    # ------------------------------------------------------------------

    def air_mass(self, soc, T_cav=None):
        """Air mass [kg] in cavern at given SOC and temperature."""
        T = self.T_cav_nom if T_cav is None else np.asarray(T_cav, dtype=float)
        s = np.clip(np.asarray(soc, dtype=float), 0.0, 1.0)
        m_max_T = self.p_max * self.V / (self.R * T)
        m_min_T = self.p_min * self.V / (self.R * T)
        return m_min_T + s * (m_max_T - m_min_T)

    def cavern_pressure(self, soc, T_cav=None):
        """Cavern pressure [Pa]."""
        T = self.T_cav_nom if T_cav is None else np.asarray(T_cav, dtype=float)
        m = self.air_mass(soc, T)
        return m * self.R * T / self.V

    def soc_from_pressure(self, pressure_Pa, T_cav=None):
        """SOC from measured cavern pressure."""
        T = self.T_cav_nom if T_cav is None else float(T_cav)
        m_max_T = self.p_max * self.V / (self.R * T)
        m_min_T = self.p_min * self.V / (self.R * T)
        m = pressure_Pa * self.V / (self.R * T)
        return np.clip((m - m_min_T) / (m_max_T - m_min_T), 0.0, 1.0)

    # ------------------------------------------------------------------
    # Power
    # ------------------------------------------------------------------

    def charge_power(self, m_dot_air, T_amb_K=None):
        """Electrical input power [kW] for compression, with T_amb correction."""
        m_dot = np.asarray(m_dot_air, dtype=float)
        w = self.specific_compression_work(T_amb_K)
        return m_dot * w / (self.eta_comp * self.eta_motor)

    def discharge_power(self, m_dot_air, T_tes_K=None):
        """
        Electrical output power [kW] from expansion, accounting for TES heat available.

        T_tes_K: current TES temperature [K]. Defaults to design temperature.
        """
        m_dot = np.asarray(m_dot_air, dtype=float)
        w_eff = self.expansion_work_effective(T_tes_K)
        return m_dot * w_eff * self.eta_exp * self.eta_gen

    def fuel_power(self, p_elec_out):
        """Fuel thermal input [kW] — always zero for A-CAES."""
        return np.zeros_like(np.asarray(p_elec_out, dtype=float))

    def tes_heat_stored_kw(self, m_dot_air, T_amb_K=None):
        """Rate of heat stored in TES during charging [kW_th]."""
        m_dot = np.asarray(m_dot_air, dtype=float)
        w = self.specific_compression_work(T_amb_K)
        return m_dot * w * self.eta_comp

    # ------------------------------------------------------------------
    # Round-trip efficiency
    # ------------------------------------------------------------------

    def round_trip_efficiency(self, T_amb_K=None, T_tes_K=None):
        """
        A-CAES RTE with T_amb and TES temperature corrections.

        eta_RT = (w_exp_eff * eta_exp * eta_gen) / (w_comp(T_amb) / (eta_comp * eta_motor))

        Physical constraint: RTE < 0.75 for A-CAES (Budt 2016).
        Typical: 0.60-0.72 at design conditions.
        """
        w_c = self.specific_compression_work(T_amb_K)
        w_e = self.expansion_work_effective(T_tes_K)
        E_out  = w_e * self.eta_exp * self.eta_gen / 3600.0       # kWh/kg
        E_in   = w_c / (self.eta_comp * self.eta_motor) / 3600.0  # kWh/kg
        return E_out / E_in

    # ------------------------------------------------------------------
    # Energy capacity
    # ------------------------------------------------------------------

    def energy_capacity_kwh(self, T_cav=None, T_tes_K=None):
        """Usable electrical energy capacity [kWh]."""
        T = self.T_cav_nom if T_cav is None else float(T_cav)
        m_max_T  = self.p_max * self.V / (self.R * T)
        m_min_T  = self.p_min * self.V / (self.R * T)
        m_usable = m_max_T - m_min_T
        w_eff = self.expansion_work_effective(T_tes_K)
        return m_usable * w_eff * self.eta_exp * self.eta_gen / 3600.0

    # ------------------------------------------------------------------
    # SOC update
    # ------------------------------------------------------------------

    def soc_update(self, soc0, power_kw, dt_hours, mode, T_amb_K=None, T_tes_K=None):
        """
        Update SOC for a single time step.
        mode: 'charge', 'discharge', or 'idle'.
        Returns new SOC clamped to [0, 1].
        """
        s = float(np.clip(soc0, 0.0, 1.0))
        P = float(power_kw)
        dt = float(dt_hours)
        if mode == "idle" or P <= 0.0 or dt <= 0.0:
            return s
        T = self.T_cav_nom
        m_max_T = self.p_max * self.V / (self.R * T)
        m_min_T = self.p_min * self.V / (self.R * T)
        m_usable = m_max_T - m_min_T
        m_cur = m_min_T + s * m_usable

        if mode == "charge":
            w_c = float(self.specific_compression_work(T_amb_K))
            E_to_air_kj = P * dt * 3600.0 * (self.eta_comp * self.eta_motor)
            m_added = E_to_air_kj / w_c
            m_new = min(m_cur + m_added, m_max_T)
        elif mode == "discharge":
            w_e = float(self.expansion_work_effective(T_tes_K))
            E_from_air_kj = P * dt * 3600.0 / (self.eta_exp * self.eta_gen)
            m_removed = E_from_air_kj / w_e
            m_new = max(m_cur - m_removed, m_min_T)
        else:
            raise ValueError(f"mode must be 'charge', 'discharge', or 'idle', got '{mode}'")
        return float(np.clip((m_new - m_min_T) / m_usable, 0.0, 1.0))
