"""
EC139 — Salinity Gradient Blue Energy (PRO) — F1a Energy Gradient Model

Pressure Retarded Osmosis (PRO) extracts work from mixing seawater and freshwater:

Gibbs free energy of mixing per m³ freshwater (theoretical maximum):
    ΔG_mix = -n_total * R * T * Σ x_i * ln(x_i)
    Simplified for dilute solutions:
    ΔG_mix ≈ R * T * (C_sw * ln(C_sw/C_mix) + C_fw * ln(C_fw/C_mix))  [J/mol → kWh/m³]

Van't Hoff osmotic pressure:
    Π = ν * R * T * ΔC_mol   [Pa]

    where ΔC_mol = (C_sw - C_fw) / M_NaCl  [mol/m³]

Net extractable energy density:
    w_gross = Π_avg * recovery_ratio          [J/m³ freshwater]
    w_net   = w_gross * eta_membrane * eta_turbine * eta_pressure_exchanger

Typical results:
    Π_seawater ~ 27 bar, ΔG_mix ~ 0.8 kWh/m³, w_net ~ 0.2-0.4 kWh/m³

References:
    Yip, N.Y. & Elimelech, M. (2012). Environ. Sci. Technol., 46, 5230-5239.
    Straub, A.P. et al. (2016). Nature Energy, 1, 16090.
    Achilli, A. & Childress, A.E. (2010). Desalination, 261, 205-211.
"""

import numpy as np

_R = 8.314      # J/(mol·K)
_RHO = 1000.0   # kg/m³ (fresh water approximation)
_J_PER_KWH = 3.6e6


class SalinityGradientPROF1a:
    """PRO salinity gradient energy — semi-empirical energy density model."""

    def __init__(self, params: dict):
        s = params["system"]
        self.C_sw         = s["C_seawater_g_per_L"]["value"]      # g/L
        self.C_fw         = s["C_freshwater_g_per_L"]["value"]     # g/L
        self.T_K          = s["T_K"]["value"]
        self.eta_membrane = s["eta_membrane"]["value"]
        self.eta_turbine  = s["eta_turbine"]["value"]
        self.eta_px       = s["eta_pressure_exchanger"]["value"]
        self.M_NaCl       = s["M_NaCl_g_per_mol"]["value"]
        self.nu           = s["nu_NaCl"]["value"]
        self.Q_feed       = s["Q_feed_m3_per_s"]["value"]
        self.recovery     = s["recovery_ratio"]["value"]

    # ------------------------------------------------------------------
    def osmotic_pressure_pa(self, C_sw_gL=None, C_fw_gL=None):
        """
        Van't Hoff osmotic pressure difference [Pa].

        Π = ν * R * T * ΔC_mol
        """
        if C_sw_gL is None: C_sw_gL = self.C_sw
        if C_fw_gL is None: C_fw_gL = self.C_fw
        C_sw = np.asarray(C_sw_gL, dtype=float)
        C_fw = np.asarray(C_fw_gL, dtype=float)
        dC_mol = (C_sw - C_fw) / self.M_NaCl * 1000.0  # mol/m³ (g/L → g/m³ / g/mol)
        return self.nu * _R * self.T_K * dC_mol  # Pa

    def gibbs_energy_kwh_per_m3(self, C_sw_gL=None, C_fw_gL=None):
        """
        Theoretical Gibbs free energy of mixing per m³ of freshwater feed [kWh/m³_fw].

        Uses the average osmotic pressure method (Yip & Elimelech 2012):
            ΔG ≈ Π_avg * V_permeate   [J per m³ freshwater]

        where Π_avg accounts for osmotic pressure decrease as freshwater permeates:
            Π_avg = Π_sw * (1 - recovery/2)

        Published theoretical value for seawater (35 g/L) vs river water (0.5 g/L):
            ~0.65-0.80 kWh/m³ of freshwater.
        """
        if C_sw_gL is None: C_sw_gL = self.C_sw
        if C_fw_gL is None: C_fw_gL = self.C_fw
        Pi = self.osmotic_pressure_pa(C_sw_gL, C_fw_gL)
        # Average pressure across permeation (Π decreases as freshwater dilutes seawater)
        Pi_avg = Pi * (1.0 - self.recovery / 2.0)
        # Energy per m³ freshwater actually permeated
        dG_J_per_m3_fw = Pi_avg  # J/m³_fw (Π [Pa] * 1 m³ = J)
        return dG_J_per_m3_fw / _J_PER_KWH  # kWh/m³_fw

    def net_energy_kwh_per_m3(self, C_sw_gL=None, C_fw_gL=None):
        """
        Net extractable energy density [kWh/m³ freshwater] after all losses.

        w_net = ΔG_theoretical * eta_membrane * eta_turbine * eta_px

        Published range: 0.2-0.4 kWh/m³ freshwater for seawater/river PRO.
        """
        dG = self.gibbs_energy_kwh_per_m3(C_sw_gL, C_fw_gL)
        return dG * self.eta_membrane * self.eta_turbine * self.eta_px

    def power_kw(self, C_sw_gL=None, C_fw_gL=None, Q_feed_m3s=None):
        """
        Electrical power output [kW].

        Q_feed is total feed (seawater + freshwater, 50:50 split).
        Freshwater volume = 0.5 * Q_feed; permeated volume = 0.5 * Q_feed * recovery.

        P [kW] = w_net [kWh/m³_fw] * Q_fw_permeated [m³/s] * 3600 [s/h]
        """
        if Q_feed_m3s is None:
            Q_feed_m3s = self.Q_feed
        Q_feed_arr = np.asarray(Q_feed_m3s, dtype=float)
        w_net = self.net_energy_kwh_per_m3(C_sw_gL, C_fw_gL)
        # Permeated freshwater flow: half of total feed (50:50 split) times recovery ratio
        Q_fw_permeated = 0.5 * Q_feed_arr * self.recovery   # m³/s freshwater permeated
        return w_net * Q_fw_permeated * 3600.0               # kWh/m³ * m³/s * 3600s/h = kW
