"""
EC208 — CO2 Geological Sequestration — F1b Injection Pressure Build-Up + Leakage Model

Extends F1a injection model with:
  1. Reservoir pressure build-up over injection lifetime.
     P_res(t) = P0 + dP_buildup(m_injected, V_pore, compressibility)
     As P_res increases toward P_max, injection rate drops (reduced driving pressure).
  2. Maximum injection pressure constraint (fracture gradient limit).
     If P_bh approaches P_max, must reduce wellhead pressure.
  3. CO2 leakage through caprock:
     m_leak = injected_total * leakage_fraction_per_year * t_years
     (linear model, conservative: IPCC 2005 quote <0.1%/yr for well-sealed)
  4. Net CO2 retained vs injected over project lifetime.
  5. Injection rate decline curve vs cumulative injection.

References:
    Nordbotten, J.M. et al. (2005). Water Resour. Res., 41(12).
    van der Meer, L.G.H. (1993). Energy Convers. Mgmt, 34, 959.
    IPCC (2005). CCS Special Report, Chapter 5.
    Benson, S. & Cole, D.R. (2008). Elements, 4, 325-331.
"""

import numpy as np

G = 9.81
MD_TO_M2 = 9.869233e-16


class CO2SequestrationF1b:
    """CO2 geological injection with reservoir pressure build-up and leakage model."""

    def __init__(self, params: dict):
        r = params["reservoir"]
        c = params["co2"]

        self.depth = r["depth_m"]["value"]
        self.thickness = r["thickness_m"]["value"]
        self.area_km2 = r["area_km2"]["value"]
        self.porosity = r["porosity"]["value"]
        self.k_mD = r["permeability_mD"]["value"]
        self.P0 = r["P_reservoir_initial_bar"]["value"] * 1e5  # Pa
        self.P_max = r["P_max_bar"]["value"] * 1e5             # Pa
        self.eff = r["storage_efficiency"]["value"]
        self.skin = r["skin_factor"]["value"]
        self.r_w = r["wellbore_radius_m"]["value"]
        self.r_e = r["drainage_radius_m"]["value"]
        self.Ct = r["compressibility"]["value"]                # 1/Pa
        self.leak_rate = r["leakage_fraction_per_year"]["value"]  # fraction/year

        self.rho_inj = c["rho_injection"]["value"]
        self.mu_inj = c["viscosity_injection"]["value"]
        self.P_wh_default = c["P_injection_wellhead"]["value"] * 1e5

    # ── Pressure build-up ─────────────────────────────────────────────────────

    def _pore_volume_m3(self):
        """Total pore volume [m3]."""
        A = self.area_km2 * 1e6
        return A * self.thickness * self.porosity

    def reservoir_pressure_pa(self, m_injected_tonnes):
        """Reservoir pressure [Pa] after injecting m_injected_tonnes tCO2.
        Simple tank model: dP = m_injected / (rho_avg * V_pore * Ct)
        rho_avg ~ rho_injection (dense CO2 at reservoir conditions)
        """
        V_pore = self._pore_volume_m3()
        m_kg = np.asarray(m_injected_tonnes, dtype=float) * 1000.0
        dP = m_kg / (self.rho_inj * V_pore * self.Ct)  # Pa
        P_res = self.P0 + dP
        return np.minimum(P_res, self.P_max)

    # ── Injection rate ────────────────────────────────────────────────────────

    def injection_rate_kg_s(self, P_wellhead_bar, m_injected_tonnes=0.0):
        """Injection rate [kg/s] accounting for reservoir pressure build-up.
        Darcy radial flow with dynamic P_res.
        """
        P_wh = np.asarray(P_wellhead_bar, dtype=float) * 1e5
        P_res = self.reservoir_pressure_pa(m_injected_tonnes)
        P_bh = P_wh + self.rho_inj * G * self.depth
        k = self.k_mD * MD_TO_M2
        ln_term = np.log(self.r_e / self.r_w) + self.skin
        dP = np.maximum(P_bh - P_res, 0.0)
        Q = (2.0 * np.pi * k * self.thickness * dP) / (self.mu_inj * ln_term)
        m_dot = Q * self.rho_inj
        return np.clip(m_dot, 0.0, None)

    def injection_rate_tco2_per_day(self, P_wellhead_bar, m_injected_tonnes=0.0):
        """Injection rate [tCO2/day]."""
        return self.injection_rate_kg_s(P_wellhead_bar, m_injected_tonnes) * 86400.0 / 1000.0

    # ── Leakage ───────────────────────────────────────────────────────────────

    def cumulative_leakage_tonnes(self, m_injected_tonnes, injection_years):
        """Cumulative CO2 leakage [tCO2] over injection period.
        m_leak = m_injected * leak_rate * t_years
        """
        m = np.asarray(m_injected_tonnes, dtype=float)
        t = np.asarray(injection_years, dtype=float)
        return m * self.leak_rate * t

    def net_retention_fraction(self, injection_years):
        """Fraction of injected CO2 retained (not leaked) after t_years."""
        t = np.asarray(injection_years, dtype=float)
        leaked_frac = self.leak_rate * t
        return np.clip(1.0 - leaked_frac, 0.0, 1.0)

    # ── Maximum injection wellhead pressure ──────────────────────────────────

    def max_wellhead_pressure_bar(self):
        """Maximum wellhead pressure before hitting fracture gradient limit [bar]."""
        # P_bh_max = P_max → P_wh_max = P_max - hydrostatic
        P_wh_max = self.P_max - self.rho_inj * G * self.depth
        return np.clip(P_wh_max / 1e5, 50.0, 300.0)

    def pressure_buildup_pct_capacity_used(self, m_injected_tonnes):
        """How full is the reservoir in terms of pressure (% of P_max-P0 reached)."""
        P_res = self.reservoir_pressure_pa(m_injected_tonnes)
        dP_current = P_res - self.P0
        dP_max = self.P_max - self.P0
        return np.clip(dP_current / dP_max * 100.0, 0.0, 100.0)

    def compute(self, P_wellhead_bar, m_injected_tonnes, injection_years):
        """Full computation returning all outputs."""
        P_res = self.reservoir_pressure_pa(m_injected_tonnes)
        inj_rate = self.injection_rate_kg_s(P_wellhead_bar, m_injected_tonnes)
        inj_tpd = self.injection_rate_tco2_per_day(P_wellhead_bar, m_injected_tonnes)
        leak = self.cumulative_leakage_tonnes(m_injected_tonnes, injection_years)
        ret_frac = self.net_retention_fraction(injection_years)
        p_pct = self.pressure_buildup_pct_capacity_used(m_injected_tonnes)
        P_wh_max = self.max_wellhead_pressure_bar()

        return {
            "reservoir_pressure_bar": P_res / 1e5,
            "injection_rate_kg_s": inj_rate,
            "injection_rate_tco2_per_day": inj_tpd,
            "cumulative_leakage_tco2": leak,
            "net_retention_fraction": ret_frac,
            "pressure_buildup_pct": p_pct,
            "max_wellhead_pressure_bar": P_wh_max,
        }
