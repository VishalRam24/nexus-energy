"""
EC133 — Tidal Lagoon — F1a Basin Model

Bidirectional basin model (ebb + flood generation):
    Energy per half-cycle: E_half = 0.5 * rho * g * A * h^2
    Generation occurs on BOTH ebb and flood:
        E_cycle = n_cycles * 0.5 * rho * g * A * h^2
    Average power: P_avg = eta * E_cycle / T_tide

Key differences from EC131 (tidal barrage):
  - Offshore enclosed lagoon (not estuary barrage)
  - Bidirectional turbines → both ebb and flood generation (n_cycles=2)
  - Slightly lower eta (~25%) due to bidirectional turbine losses
  - Smaller tidal range acceptable (lagoon can be sited at best resource, not fixed estuary)
  - Lower h_min (bidirectional turbines operate at lower head differentials)
  - Artificial lagoon wall (higher civil cost per unit area vs natural estuary)

Reference:
    Aggidis & Feather (2012), "Tidal range turbines and generation from the Severn Barrage",
    Ocean Engineering, 40, 10–17.
    Xiao et al. (2020), "Tidal Lagoon Energy Assessment", Renewable Energy.
    Tidal Lagoon Power (2015), "Swansea Bay Tidal Lagoon Scoping Report".
"""

import numpy as np


class TidalLagoonF1a:
    """Tidal lagoon bidirectional power model."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.A = u["A_lagoon"]["value"]              # m2
        self.h_design = u["h_tidal"]["value"]        # m (amplitude)
        self.T_tide = u["T_tide"]["value"]           # s
        self.eta = u["eta_plant"]["value"]           # dimensionless
        self.n_cycles = u["n_cycles_per_period"]["value"]  # 2 for bidirectional
        self.h_min = u["h_min_operation"]["value"]   # m
        self.rho = u["rho"]["value"]                 # kg/m3
        self.g = u["g"]["value"]                     # m/s2

    def theoretical_avg_power_w(self, tidal_range_m, lagoon_area_m2=None):
        """
        Theoretical average power (no efficiency losses) in Watts.
        Accounts for bidirectional generation (n_cycles cycles per period).

        Args:
            tidal_range_m: Full peak-to-trough tidal range [m]
            lagoon_area_m2: Lagoon area override [m2]
        """
        R = np.asarray(tidal_range_m, dtype=float)
        h = R / 2.0                                  # amplitude
        A = self.A if lagoon_area_m2 is None else float(lagoon_area_m2)
        # Energy per half-cycle per direction: 0.5 * rho * g * A * h^2
        # Bidirectional: multiply by n_cycles (both ebb and flood)
        E_cycle = self.n_cycles * 0.5 * self.rho * self.g * A * h ** 2
        P_theo = E_cycle / self.T_tide              # [W]
        P_theo = np.where(h < self.h_min, 0.0, P_theo)
        return np.clip(P_theo, 0.0, None)

    def avg_power_kw(self, tidal_range_m, lagoon_area_m2=None):
        """Average electrical output in kW."""
        P_w = self.theoretical_avg_power_w(tidal_range_m, lagoon_area_m2)
        return P_w * self.eta / 1000.0

    def avg_power_mw(self, tidal_range_m, lagoon_area_m2=None):
        """Average electrical output in MW."""
        return self.avg_power_kw(tidal_range_m, lagoon_area_m2) / 1000.0

    def energy_per_cycle_mwh(self, tidal_range_m, lagoon_area_m2=None):
        """Electrical energy per tidal period in MWh."""
        P_mw = self.avg_power_mw(tidal_range_m, lagoon_area_m2)
        T_h = self.T_tide / 3600.0
        return P_mw * T_h

    def capacity_factor(self, tidal_range_m, P_installed_mw, lagoon_area_m2=None):
        """Capacity factor relative to installed capacity."""
        P_mw = self.avg_power_mw(tidal_range_m, lagoon_area_m2)
        return np.clip(P_mw / P_installed_mw, 0.0, 1.0)

    def bidirectional_gain_vs_ebb_only(self):
        """
        Power gain factor of bidirectional vs ebb-only generation.
        For n_cycles=2 (both ebb+flood), theoretical factor = 2.
        But eta_lagoon is slightly lower than eta_barrage, so net gain < 2.
        """
        return self.n_cycles  # theoretical; in practice ~1.7-1.9 after penalty
