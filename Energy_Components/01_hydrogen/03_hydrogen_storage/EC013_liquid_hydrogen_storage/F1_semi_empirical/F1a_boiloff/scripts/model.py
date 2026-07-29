"""
EC013 — Liquid Hydrogen (LH2) Storage — F1a Boil-Off Model

Heat leak through tank wall:
    Q_in = U * A * (T_ambient - T_sat)         [W]

Boil-off mass rate (latent heat balance, all heat goes to phase change):
    m_dot_BO = Q_in / h_vap                     [kg/s]

Daily boil-off rate (BOR, % per day):
    BOR = (m_dot_BO * 86400 / m_LH2_initial) * 100   [%/day]

Stored mass:
    m_LH2 = rho_L * V_tank * fill_fraction      [kg]

Energy stored:
    E = m_LH2 * LHV                             [MJ]

Time to empty (assuming closed venting):
    t_empty = m_LH2 / m_dot_BO                  [s]

References:
    Sherif et al. (1997). Int. J. Hydrogen Energy, 22(7), 683-688.
    Petitpas (2018). Boil-off losses along LH2 pathway. NREL.
    Notardonato et al. (2017). IOP Conf. Series 278, 012012.
"""

import numpy as np


class LH2F1a:
    """Liquid hydrogen storage with heat-leak driven boil-off."""

    def __init__(self, params: dict):
        t = params["tank"]
        h = params["hydrogen"]
        a = params["ambient"]

        self.V_tank = t["volume"]["value"]                  # m3
        self.A_surf = t["surface_area"]["value"]            # m2
        self.m_tank = t["mass_empty"]["value"]              # kg
        self.U = t["U_overall"]["value"]                    # W/(m2.K)
        self.fill_max = t["fill_fraction_max"]["value"]

        self.T_sat = h["T_sat"]["value"]                    # K
        self.rho_L = h["rho_liquid"]["value"]               # kg/m3
        self.rho_V = h["rho_vapor"]["value"]                # kg/m3
        self.h_vap = h["h_vap"]["value"] * 1000.0           # kJ/kg -> J/kg
        self.cp_L = h["cp_liquid"]["value"] * 1000.0        # J/(kg.K)
        self.LHV = h["LHV"]["value"]                        # MJ/kg

        self.T_amb_default = a["T_ambient_default"]["value"]  # K

    # ------------------------------------------------------------------ #
    def stored_mass(self, fill_fraction):
        """Mass of LH2 stored [kg]."""
        f = np.clip(np.asarray(fill_fraction, dtype=float), 0.0, self.fill_max)
        return self.rho_L * self.V_tank * f

    def energy_stored(self, fill_fraction):
        """Stored chemical energy [MJ]."""
        return self.stored_mass(fill_fraction) * self.LHV

    def heat_leak(self, T_ambient_K=None):
        """Heat leak through tank wall [W]."""
        T_amb = self.T_amb_default if T_ambient_K is None else np.asarray(T_ambient_K, dtype=float)
        dT = T_amb - self.T_sat
        return self.U * self.A_surf * dT

    def boiloff_mass_rate(self, T_ambient_K=None):
        """Boil-off mass rate [kg/s] (steady-state, latent-heat-limited)."""
        Q = self.heat_leak(T_ambient_K)
        return Q / self.h_vap

    def boiloff_rate_percent_day(self, fill_fraction, T_ambient_K=None):
        """Daily boil-off rate as percent of stored mass [%/day]."""
        m_dot = self.boiloff_mass_rate(T_ambient_K)
        m = self.stored_mass(fill_fraction)
        safe = np.where(m > 0, m, 1.0)
        bor = np.where(m > 0, m_dot * 86400.0 / safe * 100.0, 0.0)
        return bor

    def time_to_empty_days(self, fill_fraction, T_ambient_K=None):
        """Time for tank to vent completely (no withdrawal) [days]."""
        m = self.stored_mass(fill_fraction)
        m_dot = self.boiloff_mass_rate(T_ambient_K)
        safe = np.where(m_dot > 0, m_dot, 1e-30)
        return np.where(m_dot > 0, m / safe / 86400.0, np.inf)

    def gravimetric_density(self, fill_fraction):
        """Gravimetric storage density [wt%]."""
        m = self.stored_mass(fill_fraction)
        return m / (m + self.m_tank) * 100.0

    def volumetric_density(self, fill_fraction):
        """Volumetric storage density [kg_H2/m3 of tank volume]."""
        return self.stored_mass(fill_fraction) / self.V_tank
