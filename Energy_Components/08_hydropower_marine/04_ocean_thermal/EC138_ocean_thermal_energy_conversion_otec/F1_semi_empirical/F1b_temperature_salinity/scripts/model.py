"""
EC138 — Ocean Thermal Energy Conversion (OTEC) — F1b Temperature / Salinity Model

Extends F1a efficiency curve with:
  1. Seawater density from temperature and salinity:
     rho(T, S) = 1025 + 0.8*(S - 35) - 0.15*(T - 15)  [simplified]
     Affects pump work calculations (denser water requires more pump energy).

  2. Detailed parasitic pump load breakdown:
     (a) Warm water pump:   P_pump_w = rho_w * g * Q_w * h_warm / eta_pump_warm
     (b) Cold water pump:   P_pump_c = rho_c * g * Q_c * h_cold / eta_pump_cold
     (c) Working fluid pump: P_pump_wf = wf_pump_fraction * P_gross
     Total parasitic = P_pump_w + P_pump_c + P_pump_wf

  3. Net power after detailed parasitics:
     P_net = P_gross - P_total_parasitic
     P_gross = Q_thermal * eta_gross
     Q_thermal = Q_warm * rho_w * cp_seawater * (T_warm - T_evap)  [simplified]

  4. Seasonal warm surface temperature variation:
     T_warm(phi) = T_warm_mean + seasonal_amp * cos(phi)
     where phi = 0 at summer peak, pi at winter minimum.
     Cold water temperature is constant (deep ocean, no seasonal variation).

  5. Salinity effect on osmotic pressure (minor, for physical completeness):
     Salt concentration difference between warm and cold water is negligible
     for closed-cycle OTEC but documented here for open-cycle variants.

Physics note:
  OTEC net efficiency is very low (2-3%) due to the small ΔT.
  The dominant factor is not Carnot but the large pump work for cold water
  extraction from ~1000m depth. Design optimization balances pipe diameter
  (cost) vs pump power (operational efficiency).

References:
    Vega, L.A. (2002). OTEC Primer. Mar. Technol. Soc. J., 36(4), 25-35.
    Nihous, G.C. (2007). J. Energy Resour. Technol., 129(1), 10-17.
    Faizal, M. & Ahmed, M.R. (2011). Int. J. Low-Carbon Tech., 6, 215-226.
    NREL (2020). Ocean Thermal Energy Conversion Resource Assessment.
"""

import numpy as np

_G   = 9.81       # m/s2
_CP  = 4000.0     # J/(kg*K) — seawater specific heat (approximate)
_RHO_REF = 1025.0 # kg/m3 reference


def seawater_density(T_C, S_psu):
    """
    Simplified seawater density [kg/m3].
    rho ≈ 1025 + 0.8*(S-35) - 0.15*(T-15)
    Valid over ocean operating range.
    """
    T_C   = np.asarray(T_C,   dtype=float)
    S_psu = np.asarray(S_psu, dtype=float)
    return _RHO_REF + 0.8 * (S_psu - 35.0) - 0.15 * (T_C - 15.0)


class OTECF1b:
    """OTEC closed-cycle model with detailed pump parasitics and density correction."""

    def __init__(self, params: dict):
        c = params["cycle"]
        self.T_warm_design    = c["T_warm_design_c"]["value"]
        self.T_cold_design    = c["T_cold_design_c"]["value"]
        self.eta_cycle_frac   = c["eta_cycle_fraction"]["value"]
        self.P_gross_rated    = c["P_gross_kw"]["value"]         # kW
        self.Q_warm_per_kw    = c["Q_warm_m3_per_s_per_kw"]["value"]   # m3/s/kW
        self.Q_cold_per_kw    = c["Q_cold_m3_per_s_per_kw"]["value"]
        self.eta_pump_warm    = c["eta_pump_warm"]["value"]
        self.eta_pump_cold    = c["eta_pump_cold"]["value"]
        self.eta_pump_wf      = c["eta_pump_wf"]["value"]
        self.h_warm           = c["head_warm_m"]["value"]         # m
        self.h_cold           = c["head_cold_m"]["value"]         # m
        self.wf_frac          = c["wf_pump_fraction"]["value"]
        self.S_warm           = c["salinity_warm"]["value"]       # psu
        self.S_cold           = c["salinity_cold"]["value"]       # psu
        self.T_seasonal_amp   = c["T_seasonal_amplitude_c"]["value"]

    # ------------------------------------------------------------------
    # Thermodynamic efficiency (same as F1a)
    # ------------------------------------------------------------------

    def eta_carnot(self, T_warm_c, T_cold_c):
        """Carnot efficiency for given water temperatures."""
        T_w = np.asarray(T_warm_c, dtype=float) + 273.15
        T_c = np.asarray(T_cold_c, dtype=float) + 273.15
        dT = T_w - T_c
        return np.where(dT > 0.0, 1.0 - T_c / T_w, 0.0)

    def eta_gross(self, T_warm_c, T_cold_c):
        """Gross cycle efficiency (fraction of Carnot)."""
        return self.eta_carnot(T_warm_c, T_cold_c) * self.eta_cycle_frac

    # ------------------------------------------------------------------
    # Water flows with density correction
    # ------------------------------------------------------------------

    def warm_density(self, T_warm_c=None):
        T = self.T_warm_design if T_warm_c is None else float(T_warm_c)
        return float(seawater_density(T, self.S_warm))

    def cold_density(self, T_cold_c=None):
        T = self.T_cold_design if T_cold_c is None else float(T_cold_c)
        return float(seawater_density(T, self.S_cold))

    # ------------------------------------------------------------------
    # Detailed parasitic power breakdown
    # ------------------------------------------------------------------

    def pump_power_warm_kw(self, P_gross_kw, T_warm_c=None):
        """
        Warm water pump power [kW].
        P_pump_w = rho_w * g * Q_w * h_warm / eta_pump_warm
        Q_w = Q_warm_per_kw * P_gross_kw  [m3/s]
        """
        P_gross = np.asarray(P_gross_kw, dtype=float)
        rho_w   = self.warm_density(T_warm_c)
        Q_w     = self.Q_warm_per_kw * P_gross   # m3/s
        P_pump  = rho_w * _G * Q_w * self.h_warm / self.eta_pump_warm
        return P_pump / 1000.0  # kW

    def pump_power_cold_kw(self, P_gross_kw, T_cold_c=None):
        """
        Cold water pump power [kW].
        Cold pipe is ~1000m deep: large head, dominant parasitic load.
        """
        P_gross = np.asarray(P_gross_kw, dtype=float)
        rho_c   = self.cold_density(T_cold_c)
        Q_c     = self.Q_cold_per_kw * P_gross   # m3/s
        P_pump  = rho_c * _G * Q_c * self.h_cold / self.eta_pump_cold
        return P_pump / 1000.0  # kW

    def pump_power_wf_kw(self, P_gross_kw):
        """Working fluid (ammonia) pump power [kW]."""
        return np.asarray(P_gross_kw, dtype=float) * self.wf_frac

    def total_parasitic_kw(self, P_gross_kw, T_warm_c=None, T_cold_c=None):
        """Total parasitic power [kW]."""
        return (self.pump_power_warm_kw(P_gross_kw, T_warm_c)
                + self.pump_power_cold_kw(P_gross_kw, T_cold_c)
                + self.pump_power_wf_kw(P_gross_kw))

    def parasitic_fraction(self, P_gross_kw, T_warm_c=None, T_cold_c=None):
        """Parasitic fraction of gross power."""
        P_gross = np.asarray(P_gross_kw, dtype=float)
        P_par   = self.total_parasitic_kw(P_gross, T_warm_c, T_cold_c)
        P_safe  = np.where(P_gross > 1e-6, P_gross, 1e-6)
        return P_par / P_safe

    # ------------------------------------------------------------------
    # Seasonal surface temperature
    # ------------------------------------------------------------------

    def warm_temp_seasonal(self, T_mean_c, phase_rad):
        """
        Seasonal warm surface water temperature.
        T_warm(phi) = T_mean + seasonal_amp * cos(phi)
        phi = 0 → summer peak, phi = pi → winter minimum.
        """
        T_mean = np.asarray(T_mean_c, dtype=float)
        phi    = np.asarray(phase_rad, dtype=float)
        return T_mean + self.T_seasonal_amp * np.cos(phi)

    # ------------------------------------------------------------------
    # Main power flows
    # ------------------------------------------------------------------

    def power_flows(self, T_warm_c=None, T_cold_c=None):
        """
        Compute all power flows at given sea conditions.

        Parameters
        ----------
        T_warm_c : warm surface water temperature [degC]; defaults to design
        T_cold_c : cold deep water temperature [degC]; defaults to design

        Returns
        -------
        dict: eta_carnot, eta_gross, P_gross_kw, P_net_kw, P_parasitic_kw,
              P_pump_warm_kw, P_pump_cold_kw, P_pump_wf_kw,
              parasitic_fraction, warm_density_kgm3, cold_density_kgm3
        """
        T_w = self.T_warm_design if T_warm_c is None else float(T_warm_c)
        T_c = self.T_cold_design if T_cold_c is None else float(T_cold_c)

        e_c   = float(self.eta_carnot(T_w, T_c))
        e_g   = float(self.eta_gross(T_w, T_c))

        # Gross power: scale rated P_gross by (eta_gross / eta_gross_design) to allow off-design
        e_g_design = float(self.eta_gross(self.T_warm_design, self.T_cold_design))
        ratio = e_g / max(e_g_design, 1e-9)
        P_gross = self.P_gross_rated * ratio     # kW

        # Parasitic breakdown
        P_pump_w   = float(self.pump_power_warm_kw(P_gross, T_w))
        P_pump_c   = float(self.pump_power_cold_kw(P_gross, T_c))
        P_pump_wf  = float(self.pump_power_wf_kw(P_gross))
        P_par      = P_pump_w + P_pump_c + P_pump_wf
        P_net      = max(0.0, P_gross - P_par)
        par_frac   = P_par / P_gross if P_gross > 1e-6 else 0.0

        return {
            "eta_carnot":          e_c,
            "eta_gross":           e_g,
            "P_gross_kw":          P_gross,
            "P_net_kw":            P_net,
            "P_parasitic_kw":      P_par,
            "P_pump_warm_kw":      P_pump_w,
            "P_pump_cold_kw":      P_pump_c,
            "P_pump_wf_kw":        P_pump_wf,
            "parasitic_fraction":  par_frac,
            "warm_density_kgm3":   self.warm_density(T_w),
            "cold_density_kgm3":   self.cold_density(T_c),
        }

    def eta_net(self, T_warm_c=None, T_cold_c=None):
        """Net electrical efficiency (after all parasitics)."""
        flows = self.power_flows(T_warm_c, T_cold_c)
        P_gross = flows["P_gross_kw"]
        if P_gross < 1e-6:
            return 0.0
        return flows["P_net_kw"] / P_gross * flows["eta_gross"]
