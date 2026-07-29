"""
EC124 — Liquid Air Energy Storage (LAES / CES) — F1b Thermal Model

Extends F1a (round-trip boil-off) by adding:

1. Temperature-dependent boil-off rate (BOR):
   BOR(T_amb) = BOR_ref * (1 + k_bo * (T_amb - T_ref))
   - Hotter ambient → faster heat ingress → faster evaporation
   - k_bo ≈ 0.01%/day/K (typical insulated cryogenic tank)
   - Self-discharge model: dm/dt = -BOR(T_amb) * m

2. Cold recycle effectiveness vs ambient temperature:
   eps(T_amb) = eps_ref * (1 + k_eps * (T_amb - T_ref))
   - Warmer ambient → worse cold recovery from re-gasification
   - Cold recycle reduces liquefaction work; lower eps → more w_liq needed
   - Adjusted specific liquefaction work:
       w_liq_eff(T) = w_liq_nominal * (1 - eps(T)) / (1 - eps_ref)
     (where 1-eps is fraction of work that comes from fresh compression)

3. Discharge specific work vs T_amb:
   w_disch(T_amb) = w_disch_ref * (1 + k_disch * (T_amb - T_ref))
   - Warmer ambient reduces temperature gradient in heat exchangers
   - Less work available from expansion against warm rejection

4. Full RTE including all thermal effects:
   eta_RT(T_amb) = E_out(T_amb) / E_in(T_amb)

References:
    Morgan, R. et al. (2015). Applied Energy, 137, 845-853.
    Sciacovelli, A. et al. (2017). Applied Energy, 190, 84-98.
    Guizzi, G.L. et al. (2015). Energy, 93(1), 1382-1394.
"""

import numpy as np


class LAESF1b:
    """Liquid Air Energy Storage — thermal model with T_amb effects and cold recycle."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.w_liq_ref = u["specific_liquefaction_kwh_per_kg"]["value"]      # kWh/kg
        self.w_disch_ref = u["specific_discharge_kwh_per_kg"]["value"]        # kWh/kg
        self.eta_liq = u["eta_liquefier"]["value"]
        self.eta_pump = u["eta_pump"]["value"]
        self.eta_exp = u["eta_expander"]["value"]
        self.eta_gen = u["eta_generator"]["value"]
        self.m_tank_max = u["tank_capacity_kg"]["value"]                       # kg
        self.T_storage = u["T_storage"]["value"]                               # K
        self.bor_ref = u["boil_off_per_day_ref"]["value"]                      # fraction/day
        self.T_amb_ref = u["T_amb_ref"]["value"]                               # K
        self.rho_liquid = u["rho_liquid_air"]["value"]                         # kg/m3
        self.k_bo = u["boil_off_T_coeff"]["value"]                             # 1/K
        self.eps_ref = u["cold_recycle_eff_ref"]["value"]                      # dimensionless
        self.k_eps = u["cold_recycle_T_coeff"]["value"]                        # 1/K
        self.k_disch = u["discharge_work_T_coeff"]["value"]                    # 1/K

    # ------------------------------------------------------------------
    # Boil-off rate: T_amb dependent
    # ------------------------------------------------------------------

    def boil_off_rate_per_day(self, T_amb_K=None):
        """
        Boil-off rate [fraction/day] at given ambient temperature.

        BOR(T_amb) = BOR_ref * (1 + k_bo * (T_amb - T_ref))

        Hotter ambient → faster heat ingress into insulated tank.

        Args:
            T_amb_K: Ambient temperature [K] (default: T_amb_ref)
        """
        if T_amb_K is None:
            return self.bor_ref
        T = np.asarray(T_amb_K, dtype=float)
        return self.bor_ref * (1.0 + self.k_bo * (T - self.T_amb_ref))

    def k_bo_per_hour(self, T_amb_K=None):
        """Boil-off rate constant [1/h] = BOR_per_day / 24."""
        return self.boil_off_rate_per_day(T_amb_K) / 24.0

    # ------------------------------------------------------------------
    # Cold recycle effectiveness: T_amb dependent
    # ------------------------------------------------------------------

    def cold_recycle_effectiveness(self, T_amb_K=None):
        """
        Cold recycle effectiveness [-] at given ambient temperature.

        eps(T_amb) = eps_ref * (1 + k_eps * (T_amb - T_ref))

        Warmer ambient degrades cold recovery from re-gasification process.

        Args:
            T_amb_K: Ambient temperature [K] (default: T_amb_ref)
        """
        if T_amb_K is None:
            return self.eps_ref
        T = np.asarray(T_amb_K, dtype=float)
        eps = self.eps_ref * (1.0 + self.k_eps * (T - self.T_amb_ref))
        return np.clip(eps, 0.0, 1.0)

    def effective_liquefaction_work(self, T_amb_K=None):
        """
        Effective specific liquefaction work [kWh/kg] corrected for cold recycle.

        Higher eps → better cold recovery → less net work required:
            w_liq_eff = w_liq_ref * (1 - eps) / (1 - eps_ref)

        Args:
            T_amb_K: Ambient temperature [K]
        """
        eps = self.cold_recycle_effectiveness(T_amb_K)
        denom = max(1.0 - self.eps_ref, 1e-6)
        return self.w_liq_ref * (1.0 - eps) / denom

    # ------------------------------------------------------------------
    # Discharge specific work: T_amb dependent
    # ------------------------------------------------------------------

    def effective_discharge_work(self, T_amb_K=None):
        """
        Effective specific discharge work [kWh/kg] at given T_amb.

        w_disch(T_amb) = w_disch_ref * (1 + k_disch * (T_amb - T_ref))

        Warmer ambient reduces effective expansion work.

        Args:
            T_amb_K: Ambient temperature [K]
        """
        if T_amb_K is None:
            return self.w_disch_ref
        T = np.asarray(T_amb_K, dtype=float)
        w = self.w_disch_ref * (1.0 + self.k_disch * (T - self.T_amb_ref))
        return np.maximum(w, 0.0)

    # ------------------------------------------------------------------
    # Round-trip efficiency with all thermal effects
    # ------------------------------------------------------------------

    def round_trip_efficiency(self, T_amb_K=None):
        """
        RTE including cold-recycle and T_amb effects:
            E_out = w_disch(T) * eta_pump * eta_exp * eta_gen
            E_in  = w_liq_eff(T) / eta_liq
            eta_RT = E_out / E_in
        """
        w_d = self.effective_discharge_work(T_amb_K)
        w_l = self.effective_liquefaction_work(T_amb_K)
        E_out = w_d * self.eta_pump * self.eta_exp * self.eta_gen
        E_in = w_l / self.eta_liq
        return E_out / E_in

    # ------------------------------------------------------------------
    # Power
    # ------------------------------------------------------------------

    def charge_power(self, m_dot_liquid, T_amb_K=None):
        """Electrical input power [kW] to liquefy m_dot [kg/s]."""
        m_dot = np.asarray(m_dot_liquid, dtype=float)
        w_l = self.effective_liquefaction_work(T_amb_K)
        return m_dot * w_l * 3600.0 / self.eta_liq

    def discharge_power(self, m_dot_liquid, T_amb_K=None):
        """Electrical output power [kW] from expanding m_dot [kg/s]."""
        m_dot = np.asarray(m_dot_liquid, dtype=float)
        w_d = self.effective_discharge_work(T_amb_K)
        return m_dot * w_d * 3600.0 * self.eta_pump * self.eta_exp * self.eta_gen

    # ------------------------------------------------------------------
    # Storage state
    # ------------------------------------------------------------------

    def liquid_mass(self, soc):
        """Liquid air mass [kg]."""
        s = np.clip(np.asarray(soc, dtype=float), 0.0, 1.0)
        return s * self.m_tank_max

    def soc_from_mass(self, mass):
        m = np.asarray(mass, dtype=float)
        return np.clip(m / self.m_tank_max, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Boil-off self-discharge
    # ------------------------------------------------------------------

    def boil_off_mass(self, soc, time_hours, T_amb_K=None):
        """
        Liquid mass remaining [kg] after time_hours of storage.

        m(t) = m0 * exp(-k_bo_per_h(T_amb) * t)

        Args:
            soc:        Initial SOC [0-1]
            time_hours: Storage duration [h]
            T_amb_K:    Ambient temperature [K]
        """
        m0 = self.liquid_mass(soc)
        t = np.asarray(time_hours, dtype=float)
        k = self.k_bo_per_hour(T_amb_K)
        return m0 * np.exp(-k * t)

    def soc_after_standby(self, soc, time_hours, T_amb_K=None):
        """SOC after boil-off during standby."""
        return self.soc_from_mass(self.boil_off_mass(soc, time_hours, T_amb_K))

    def boil_off_loss_fraction(self, time_hours, T_amb_K=None):
        """Fraction of liquid lost to boil-off over time_hours."""
        t = np.asarray(time_hours, dtype=float)
        k = self.k_bo_per_hour(T_amb_K)
        return 1.0 - np.exp(-k * t)

    # ------------------------------------------------------------------
    # Energy capacity
    # ------------------------------------------------------------------

    def energy_capacity_kwh(self, T_amb_K=None):
        """Usable electrical energy capacity [kWh] (full tank, at T_amb)."""
        w_d = self.effective_discharge_work(T_amb_K)
        return self.m_tank_max * w_d * self.eta_pump * self.eta_exp * self.eta_gen
