"""
EC133 — Tidal Lagoon — F1b Tidal Range / Efficiency Model

Extends F1a basin model with:
  1. Turbine efficiency vs head ratio:
     eta(h) = eta_peak * (1 - k_h * (h/h_design - 1)^2)
     At low head (neap tide), bidirectional turbine efficiency degrades.
     Returns 0 below h_min.

  2. Seawater density correction (temperature and salinity):
     rho(T, S) = rho_ref + 0.8*(S - S_ref) - 0.15*(T - T_ref)
     Coastal lagoon experiences seasonal and tidal variation in T/S.

  3. Spring-neap tidal cycle:
     Monthly variation: tidal range cycles between spring (~+30%) and neap (~-30%)
     relative to mean tidal range. Can pass variable tidal_range directly or use
     spring_neap_amplitude parameter.

  4. Pumping mode gain:
     Optional: pumping water into lagoon at high tide adds potential energy.
     Net gain = pumping_gain_fraction of ebb generation energy.

References:
    Aggidis, G.A. & Feather, O. (2012). Tidal range turbines and generation
        from the Severn Barrage. Ocean Engineering, 40, 10–17.
    Baker, A.C. (1991). Tidal Power. Peter Peregrinus, IEE.
    Xiao, Q. et al. (2020). Tidal Lagoon Energy Assessment, Renew. Energy.
    Tidal Lagoon Power (2015). Swansea Bay Tidal Lagoon Scoping Report.
"""

import numpy as np


class TidalLagoonF1b:
    """Tidal lagoon — efficiency vs head, density correction, spring-neap, pumping."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.A          = u["A_lagoon"]["value"]
        self.h_design   = u["h_tidal"]["value"]
        self.T_tide     = u["T_tide"]["value"]
        self.eta_peak   = u["eta_peak"]["value"]
        self.h_min      = u["h_min_operation"]["value"]
        self.rho_ref    = u["rho_ref"]["value"]
        self.g          = u["g"]["value"]
        self.k_h        = u["k_head_efficiency"]["value"]
        self.S_ref      = u["S_ref"]["value"]
        self.T_ref      = u["T_ref"]["value"]
        self.n_cycles   = u["n_cycles_per_period"]["value"]
        self.pump_frac  = u["pumping_gain_fraction"]["value"]
        self.sn_amp     = u["spring_neap_amplitude"]["value"]

    # ------------------------------------------------------------------
    # Density
    # ------------------------------------------------------------------

    def seawater_density(self, T_C=None, S_psu=None):
        """
        Seawater density [kg/m3] as function of temperature and salinity.
        rho ≈ rho_ref + 0.8*(S - S_ref) - 0.15*(T - T_ref)
        """
        T = self.T_ref if T_C is None else np.asarray(T_C, dtype=float)
        S = self.S_ref if S_psu is None else np.asarray(S_psu, dtype=float)
        return self.rho_ref + 0.8 * (S - self.S_ref) - 0.15 * (T - self.T_ref)

    # ------------------------------------------------------------------
    # Turbine efficiency vs head
    # ------------------------------------------------------------------

    def turbine_efficiency(self, tidal_amplitude_m):
        """
        Turbine efficiency as function of tidal amplitude (half-range).

        eta(h) = eta_peak * (1 - k_h * (h/h_design - 1)^2)

        Physical interpretation:
        - At h = h_design: eta = eta_peak (design point)
        - Below design head (neap tide): efficiency degrades quadratically
        - Above design head (spring tide): slight degradation (overflow / speed limits)
        - Below h_min: turbine cannot operate → eta = 0
        """
        h = np.asarray(tidal_amplitude_m, dtype=float)
        h_ratio = h / self.h_design
        eta = self.eta_peak * (1.0 - self.k_h * (h_ratio - 1.0) ** 2)
        eta = np.where(h < self.h_min, 0.0, eta)
        return np.clip(eta, 0.0, self.eta_peak)

    # ------------------------------------------------------------------
    # Spring-neap cycle
    # ------------------------------------------------------------------

    def spring_neap_range(self, tide_phase_rad=None, mean_range_m=None):
        """
        Tidal range as function of spring-neap cycle phase.

        tidal_range(phi) = mean_range * (1 + sn_amp * cos(phi))

        where phi = 0 at spring tide, pi at neap tide.

        Parameters
        ----------
        tide_phase_rad : float or array [rad], 0 = spring, pi = neap
        mean_range_m   : mean peak-to-trough tidal range [m]; if None, use 2*h_design

        Returns: tidal range [m]
        """
        mean_R = 2.0 * self.h_design if mean_range_m is None else float(mean_range_m)
        if tide_phase_rad is None:
            return mean_R
        phi = np.asarray(tide_phase_rad, dtype=float)
        return mean_R * (1.0 + self.sn_amp * np.cos(phi))

    # ------------------------------------------------------------------
    # Power output
    # ------------------------------------------------------------------

    def theoretical_avg_power_w(self, tidal_range_m, lagoon_area_m2=None, T_C=None, S_psu=None):
        """
        Theoretical average power [W] (before turbine/generator losses).
        P_theo = n_cycles * 0.5 * rho * g * A * h^2 / T_tide
        where h = tidal_range / 2.
        """
        R   = np.asarray(tidal_range_m, dtype=float)
        h   = R / 2.0
        A   = self.A if lagoon_area_m2 is None else float(lagoon_area_m2)
        rho = self.seawater_density(T_C, S_psu)

        E_cycle = self.n_cycles * 0.5 * rho * self.g * A * h ** 2
        P_theo  = E_cycle / self.T_tide
        P_theo  = np.where(h < self.h_min, 0.0, P_theo)
        return np.clip(P_theo, 0.0, None)

    def avg_power_kw(self, tidal_range_m, lagoon_area_m2=None, T_C=None, S_psu=None,
                     pumping_mode=False):
        """
        Average electrical output [kW] with all F1b corrections.

        Corrections applied:
          (1) Turbine efficiency vs head: degrades at neap tide
          (2) Density correction: rho(T,S) affects theoretical power
          (3) Pumping mode: optional +7% gain
        """
        h   = np.asarray(tidal_range_m, dtype=float) / 2.0
        P_w = self.theoretical_avg_power_w(tidal_range_m, lagoon_area_m2, T_C, S_psu)
        eta = self.turbine_efficiency(h)

        P_kw = P_w * eta / 1000.0
        if pumping_mode:
            P_kw = P_kw * (1.0 + self.pump_frac)
        return P_kw

    def avg_power_mw(self, tidal_range_m, lagoon_area_m2=None, T_C=None, S_psu=None,
                     pumping_mode=False):
        return self.avg_power_kw(tidal_range_m, lagoon_area_m2, T_C, S_psu, pumping_mode) / 1000.0

    def energy_per_cycle_mwh(self, tidal_range_m, lagoon_area_m2=None, T_C=None, S_psu=None):
        P_mw = self.avg_power_mw(tidal_range_m, lagoon_area_m2, T_C, S_psu)
        T_h  = self.T_tide / 3600.0
        return P_mw * T_h

    def capacity_factor(self, tidal_range_m, P_installed_mw, lagoon_area_m2=None,
                        T_C=None, S_psu=None):
        P_mw = self.avg_power_mw(tidal_range_m, lagoon_area_m2, T_C, S_psu)
        return np.clip(P_mw / P_installed_mw, 0.0, 1.0)

    def density_effect_pct(self, T_C, S_psu):
        """Density change relative to reference [%]."""
        rho = self.seawater_density(T_C, S_psu)
        return (rho - self.rho_ref) / self.rho_ref * 100.0
