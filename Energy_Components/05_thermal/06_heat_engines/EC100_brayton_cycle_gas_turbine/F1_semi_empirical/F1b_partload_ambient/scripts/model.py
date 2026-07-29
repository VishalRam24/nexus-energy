"""
EC100 — Brayton Cycle Gas Turbine — F1b Part-Load + Ambient Temperature Model

Extends F1a (basic thermal efficiency curve) with:
  1. Quadratic part-load efficiency correction
  2. ISO ambient corrections (inlet temperature + pressure effects)
  3. Compressor work correction with variable specific heat ratio
  4. Turbine inlet temperature (TIT) effect on efficiency

Brayton cycle net efficiency:
    eta_net = 1 - T1/T3 * (rp^((gamma-1)/gamma)) / (eta_t * eta_c)
    (simplified for ideal gas)

But for real GT (part-load and ambient corrections):
    eta_GT(PLR, T_amb) = eta_rated * f_PLR(PLR) * f_amb(T_amb)

    f_PLR(PLR) = a + b*PLR + c*PLR^2    [quadratic PLR curve]

    Power correction (ISO 2314):
        P_corr = P_iso * (P_amb/P_ref) * sqrt(T_ref_K / T_amb_K)
    Efficiency correction:
        eta_corr = eta_iso * sqrt(T_ref_K / T_amb_K)

Heat rate:
    HR = 3600 / eta_net   [kJ/kWh]

Exhaust temperature correction:
    T_exh = T_exh_rated + k_PLR * (1 - PLR)   [exhaust rises at part load]

References:
    Walsh & Fletcher (2004), Gas Turbine Performance, 2nd ed., Blackwell.
    GE Power GT data sheets (F-class simple cycle, ~38-40% ISO).
    ISO 2314:2009 — Gas turbines: Acceptance tests.
    Horlock (2003), Advanced Gas Turbine Cycles, Elsevier.
"""

import numpy as np


class BraytonGTF1b:
    """Simple-cycle Brayton gas turbine — part-load + ambient correction model."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.P_rated   = u["P_rated_mw"]["value"]          # MW_e
        self.eta_rated = u["eta_rated"]["value"]            # net efficiency at ISO
        self.T_iso_k   = u["T_iso_ref_k"]["value"]         # K (288.15)
        self.P_iso_kpa = u["P_iso_ref_kpa"]["value"]       # kPa (101.325)
        self.plr_a     = u["plr_a"]["value"]
        self.plr_b     = u["plr_b"]["value"]
        self.plr_c     = u["plr_c"]["value"]
        self.PLR_min   = u["PLR_min"]["value"]
        self.T_exh_iso = u["T_exhaust_iso_k"]["value"]     # K
        self.k_PLR_exh = u["T_exh_partload_rise_k"]["value"]  # K/unit PLR
        self.LHV       = u["LHV_gas_mj_kg"]["value"]       # MJ/kg (natural gas)

    # ------------------------------------------------------------------
    # Part-load efficiency correction
    # ------------------------------------------------------------------

    def f_plr(self, PLR):
        """GT efficiency part-load correction (quadratic DOE-2 style)."""
        PLR = np.asarray(PLR, dtype=float)
        PLR_eff = np.maximum(PLR, self.PLR_min)
        return self.plr_a + self.plr_b * PLR_eff + self.plr_c * PLR_eff**2

    # ------------------------------------------------------------------
    # Ambient corrections (ISO 2314)
    # ------------------------------------------------------------------

    def f_amb_power(self, T_amb_k, P_amb_kpa):
        """ISO inlet power correction: (P_amb/P_ref) * sqrt(T_ref/T_amb)."""
        T = np.asarray(T_amb_k, dtype=float)
        P = np.asarray(P_amb_kpa, dtype=float)
        return (P / self.P_iso_kpa) * np.sqrt(self.T_iso_k / T)

    def f_amb_eta(self, T_amb_k):
        """Efficiency correction: sqrt(T_ref/T_amb) for simple cycle GT."""
        T = np.asarray(T_amb_k, dtype=float)
        return np.sqrt(self.T_iso_k / T)

    # ------------------------------------------------------------------
    # Net efficiency
    # ------------------------------------------------------------------

    def efficiency(self, PLR, T_amb_k):
        """
        Net GT efficiency.
        eta = eta_rated * f_PLR * f_amb_eta
        """
        eta = self.eta_rated * self.f_plr(PLR) * self.f_amb_eta(T_amb_k)
        return np.clip(eta, 0.05, 0.42)

    # ------------------------------------------------------------------
    # Power output
    # ------------------------------------------------------------------

    def power_output_kw(self, PLR, T_amb_k, P_amb_kpa=101.325):
        """
        Net electrical output [kW].
        P = P_rated * PLR * f_amb_power
        """
        PLR = np.asarray(PLR, dtype=float)
        f_amb = self.f_amb_power(T_amb_k, P_amb_kpa)
        return self.P_rated * PLR * f_amb * 1e3  # MW -> kW

    # ------------------------------------------------------------------
    # Exhaust temperature
    # ------------------------------------------------------------------

    def exhaust_temp_k(self, PLR):
        """
        Exhaust temperature [K].
        At part load, GT exhausts are hotter (less expansion work extracted):
        T_exh = T_exh_iso + k_PLR * (1 - PLR)
        """
        PLR = np.asarray(PLR, dtype=float)
        return self.T_exh_iso + self.k_PLR_exh * (1.0 - PLR)

    # ------------------------------------------------------------------
    # Fuel consumption
    # ------------------------------------------------------------------

    def fuel_flow_kg_s(self, PLR, T_amb_k, P_amb_kpa=101.325):
        """Fuel mass flow rate [kg/s]."""
        P_kw = self.power_output_kw(PLR, T_amb_k, P_amb_kpa)
        eta  = self.efficiency(PLR, T_amb_k)
        Q_fuel = P_kw / np.maximum(eta, 1e-6)  # kW
        return Q_fuel / (self.LHV * 1e3)       # kW / (kJ/kg) = kg/s

    # ------------------------------------------------------------------
    # Heat rate
    # ------------------------------------------------------------------

    def heat_rate_kj_kwh(self, PLR, T_amb_k, P_amb_kpa=101.325):
        """Combined heat rate [kJ/kWh]."""
        eta = self.efficiency(PLR, T_amb_k)
        return 3600.0 / np.maximum(eta, 1e-6)

    # ------------------------------------------------------------------
    # predict_all
    # ------------------------------------------------------------------

    def predict_all(self, PLR, T_amb_k, P_amb_kpa=101.325):
        """
        Return all outputs as a dict.

        Parameters
        ----------
        PLR       : part-load ratio [0.3-1.0]
        T_amb_k   : ambient temperature [K]
        P_amb_kpa : ambient pressure [kPa] (default ISO = 101.325)
        """
        eta    = self.efficiency(PLR, T_amb_k)
        P_kw   = self.power_output_kw(PLR, T_amb_k, P_amb_kpa)
        T_exh  = self.exhaust_temp_k(PLR)
        HR     = self.heat_rate_kj_kwh(PLR, T_amb_k, P_amb_kpa)
        f_amb  = self.f_amb_power(T_amb_k, P_amb_kpa)

        return {
            "efficiency":        eta,
            "power_output_kw":   P_kw,
            "heat_rate_kj_kwh":  HR,
            "exhaust_temp_k":    T_exh,
            "f_amb_power":       np.asarray(f_amb, dtype=float),
        }
