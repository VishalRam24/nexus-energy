"""
EC101 — Combined Cycle Gas Turbine (CCGT) — F1a Efficiency Curve

eta(PLR, T_amb) = eta_iso * f_PLR(PLR) * f_amb(T_amb)
  f_PLR = a0 + a1*PLR + a2*PLR^2    (part-load correction)
  f_amb = 1 - k_amb*(T_amb - T_ref) (ambient temperature derating)

P_out        = P_rated * PLR
fuel_rate    = P_out / eta            [MW_th] -> [kg/s] via LHV
exhaust_temp = 80 + 520 * PLR^0.3    [degC]   (empirical exhaust model)

Reference:
    Kehlhofer, R., Hannemann, F., Stirnimann, F., Rukes, B. (2009).
    Combined-Cycle Gas & Steam Turbine Power Plants, 3rd ed.
    PennWell Corporation.
"""

import numpy as np


class CCGTF1a:
    """CCGT efficiency as a function of part-load ratio and ambient temperature."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.P_rated      = u["rated_power_mw"]["value"]          # MW
        self.eta_iso      = u["eta_iso"]["value"]                  # dimensionless
        self.T_amb_ref    = u["T_amb_ref"]["value"]                # degC
        self.k_amb        = u["k_amb"]["value"]                    # 1/K
        self.a0           = u["plr_coeffs"]["a0"]["value"]
        self.a1           = u["plr_coeffs"]["a1"]["value"]
        self.a2           = u["plr_coeffs"]["a2"]["value"]
        self.LHV_gas      = u["LHV_gas"]["value"]                  # MJ/kg

    # ------------------------------------------------------------------
    # Correction factors
    # ------------------------------------------------------------------

    def f_plr(self, plr):
        """Quadratic part-load efficiency correction factor."""
        plr = np.asarray(plr, dtype=float)
        return self.a0 + self.a1 * plr + self.a2 * plr ** 2

    def f_amb(self, T_amb):
        """Linear ambient-temperature efficiency derating factor."""
        T_amb = np.asarray(T_amb, dtype=float)
        return 1.0 - self.k_amb * (T_amb - self.T_amb_ref)

    # ------------------------------------------------------------------
    # Primary outputs
    # ------------------------------------------------------------------

    def efficiency(self, plr, T_amb):
        """Net LHV efficiency (dimensionless)."""
        return self.eta_iso * self.f_plr(plr) * self.f_amb(T_amb)

    def power_mw(self, plr):
        """Electrical output power [MW]."""
        return self.P_rated * np.asarray(plr, dtype=float)

    def fuel_rate_kgs(self, plr, T_amb):
        """Natural gas mass flow rate [kg/s]."""
        P_out = self.power_mw(plr)               # MW_e
        eta   = self.efficiency(plr, T_amb)
        # Avoid division by zero at eta=0
        eta_safe = np.where(np.asarray(eta) > 1e-6, eta, 1e-6)
        fuel_mw  = P_out / eta_safe              # MW_th
        return fuel_mw / self.LHV_gas            # kg/s  (MW / MJ/kg = t/s -> *1000 -> kg/s)
        # 1 MW / (MJ/kg) = 1 (MJ/s) / (MJ/kg) = 1 kg/s  ✓

    def exhaust_temp_c(self, plr):
        """Stack exhaust temperature [degC] — empirical correlation."""
        plr = np.asarray(plr, dtype=float)
        return 80.0 + 520.0 * plr ** 0.3
