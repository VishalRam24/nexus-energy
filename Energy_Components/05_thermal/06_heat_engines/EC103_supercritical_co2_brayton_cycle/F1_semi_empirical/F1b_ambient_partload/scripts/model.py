"""
EC103 — Supercritical CO2 Brayton Cycle — F1b T_reject Ambient + Part-Load + Recuperator

Extends F1a with:
  1. T_reject (compressor inlet) temperature effect on efficiency:
     - Critical near CO2 critical temperature (31.1 degC)
     - Within 2 K of critical: nonlinear penalty because real-gas effects
       cause anomalous compressibility — compressor work shoots up
     - Beyond design: linear derating k_T_reject [1/K]
  2. Recuperator effectiveness degrades at part-load:
     - Lower mass flow at part-load changes HTX NTU response
     - eps_recup = eps_design * (1 + slope*(1-PLR))  [slope < 0]
     - Lower effectiveness reduces heat recovery => higher fuel input
  3. Part-load efficiency curve: quadratic (relatively flat for sCO2)

Physical background:
    sCO2 efficiency is extremely sensitive to T_reject because:
    a) CO2 near critical point has cp -> infinity, rho -> infinity
       This makes compression work near-isothermal if T_reject < T_critical
       But above critical point cp drops sharply — compressor work rises
    b) The recuperator is critical for sCO2 efficiency because the
       cycle temperature spread is narrow (~300-700 degC range);
       without recuperation, efficiency drops from ~45% to ~20%

Combined efficiency model:
    eta_carnot_limit = 1 - (T_reject_K) / (T_in_K)
    eta_cycle = eta_rated * f_PLR * f_T_reject * f_recuperator
    Capped at: eta_carnot_limit

    f_PLR = 1 - a_partload*(1-PLR)^2
    f_T_reject = 1 - k_T * max(0, T_reject - T_reject_design) * (1 + penalty_near_critical)
    f_recup = eta_recup_effective / eta_recup_design  (penalty from reduced recuperation)

References:
    Dostal, V., Driscoll, M.J. & Hejzlar, P. (2004) MIT-ANP-TR-100.
    Crespi, F. et al. (2017) Appl. Energy 195, 152-183.
    Wright, S.A. et al. (2010) SANDIA Report SAND2010-0171.
    Sarkar, J. & Bhattacharyya, S. (2009) Energy Conv. Manag. 50, 1991-1997.
"""

import numpy as np

_T_CRIT_CO2 = 31.1   # degC
_P_CRIT_CO2 = 7.38   # MPa


class SCO2BraytonF1b:
    """sCO2 Brayton cycle with critical-point T_reject penalty + recuperator degradation."""

    def __init__(self, params: dict):
        c = params["cycle"]
        self.P_rated       = float(c["P_rated_w"]["value"])
        self.eta_rated     = float(c["eta_rated"]["value"])
        self.a_partload    = float(c["a_partload"]["value"])
        self.T_in_design   = float(c["T_in_design_c"]["value"])
        self.T_rej_design  = float(c["T_reject_design_c"]["value"])
        self.k_T_reject    = float(c["k_T_reject"]["value"])
        self.T_crit_band   = float(c["T_critical_penalty_c"]["value"])
        self.eps_recup_des = float(c["eps_recup_design"]["value"])
        self.eps_recup_slp = float(c["eps_recup_PLR_slope"]["value"])
        self.PLR_min       = float(c["PLR_min"]["value"])
        self.aux_frac      = float(c["aux_fraction"]["value"])

    # ------------------------------------------------------------------
    # Correction factors
    # ------------------------------------------------------------------

    def eta_carnot_limit(self, T_in_c, T_reject_c):
        """Carnot efficiency upper bound."""
        T_h = np.asarray(T_in_c, dtype=float) + 273.15
        T_c = np.asarray(T_reject_c, dtype=float) + 273.15
        return np.clip(1.0 - T_c / T_h, 0.0, 1.0)

    def f_partload(self, PLR):
        """Part-load correction (quadratic); sCO2 is relatively flat."""
        PLR = np.asarray(PLR, dtype=float)
        return 1.0 - self.a_partload * (1.0 - PLR) ** 2

    def f_T_reject(self, T_reject_c):
        """
        T_reject correction factor.

        Two regimes:
        1. T_reject <= T_reject_design:
           f = 1 + (small benefit from cooler reject — compressor work reduced)
           BUT below T_critical+margin: extra penalty (density anomaly causes
           control issues in practice — we conservatively cap benefit at 0)

        2. T_reject > T_reject_design:
           f = 1 - k_T_reject * (T_reject - T_rej_design)
           With nonlinear amplification near critical:
           if T_reject in [T_crit, T_crit + band]: multiply penalty by 1.5
           (represents near-critical compressor work increase)
        """
        T = np.asarray(T_reject_c, dtype=float)
        dT = T - self.T_rej_design

        # Base derating (only when above design)
        base_factor = np.where(dT > 0, 1.0 - self.k_T_reject * dT, 1.0)

        # Near-critical nonlinear penalty (for T_reject near T_crit)
        # If T_reject is within T_crit_band above T_critical:
        near_crit = (T > _T_CRIT_CO2) & (T < _T_CRIT_CO2 + self.T_crit_band)
        # Near-critical: compressor working in anomalous region => extra 50% penalty
        near_crit_amplifier = np.where(near_crit, 1.5, 1.0)

        # Apply amplifier only to the derating (not to the boost)
        extra_dT = np.where(dT > 0, dT, 0.0)
        f = 1.0 - self.k_T_reject * extra_dT * near_crit_amplifier

        return np.clip(f, 0.1, 1.10)

    def recuperator_effectiveness(self, PLR):
        """
        Recuperator effectiveness at part-load.
        eps = eps_design * (1 + slope*(1-PLR))
        slope < 0 => eps drops at part load.
        Physical basis: lower mass flow changes HX duty vs UA balance (NTU method).
        """
        PLR = np.asarray(PLR, dtype=float)
        eps = self.eps_recup_des * (1.0 + self.eps_recup_slp * (1.0 - PLR))
        return np.clip(eps, 0.50, 1.0)

    def f_recuperator(self, PLR):
        """
        Recuperator effectiveness penalty factor.
        Relative to design: f = eps(PLR) / eps_design.
        Impact on cycle efficiency is roughly proportional to recuperation fraction.
        (At PLR=1: f=1. At PLR=0.3: f ~ 1 + slope*0.7.)
        """
        eps = self.recuperator_effectiveness(PLR)
        return eps / self.eps_recup_des

    # ------------------------------------------------------------------
    # Efficiency
    # ------------------------------------------------------------------

    def efficiency_gross(self, PLR, T_in_c=None, T_reject_c=None):
        """
        Gross cycle efficiency.
        eta = eta_rated * f_PLR * f_T_reject * f_recup, capped at Carnot.
        """
        PLR = np.asarray(PLR, dtype=float)
        T_in  = self.T_in_design  if T_in_c     is None else T_in_c
        T_rej = self.T_rej_design if T_reject_c is None else T_reject_c
        T_in  = np.asarray(T_in,  dtype=float)
        T_rej = np.asarray(T_rej, dtype=float)

        eta = (self.eta_rated
               * self.f_partload(PLR)
               * self.f_T_reject(T_rej)
               * self.f_recuperator(PLR))

        eta_c = self.eta_carnot_limit(T_in, T_rej)
        eta = np.where(PLR < self.PLR_min, 0.0, eta)
        return np.clip(np.minimum(eta, eta_c), 0.0, 1.0)

    def efficiency_net(self, PLR, T_in_c=None, T_reject_c=None):
        return self.efficiency_gross(PLR, T_in_c, T_reject_c) * (1.0 - self.aux_frac)

    # ------------------------------------------------------------------
    # Power and heat flows
    # ------------------------------------------------------------------

    def power_output_w(self, PLR):
        PLR = np.asarray(PLR, dtype=float)
        return np.where(PLR >= self.PLR_min, PLR * self.P_rated, 0.0)

    def heat_input_w(self, PLR, T_in_c=None, T_reject_c=None):
        P_net = self.power_output_w(PLR)
        eta = self.efficiency_net(PLR, T_in_c, T_reject_c)
        safe = np.where(eta > 0.001, eta, np.inf)
        return P_net / safe

    def heat_rejection_w(self, PLR, T_in_c=None, T_reject_c=None):
        Q_in  = self.heat_input_w(PLR, T_in_c, T_reject_c)
        P_net = self.power_output_w(PLR)
        return np.maximum(Q_in - P_net, 0.0)

    # ------------------------------------------------------------------
    # Full evaluation
    # ------------------------------------------------------------------

    def evaluate(self, PLR, T_in_c=None, T_reject_c=None):
        PLR = np.asarray(PLR, dtype=float)
        T_in  = self.T_in_design  if T_in_c     is None else T_in_c
        T_rej = self.T_rej_design if T_reject_c is None else T_reject_c
        T_in  = np.asarray(T_in,  dtype=float) * np.ones_like(PLR)
        T_rej = np.asarray(T_rej, dtype=float) * np.ones_like(PLR)
        return {
            "efficiency_gross":        self.efficiency_gross(PLR, T_in, T_rej),
            "efficiency_net":          self.efficiency_net(PLR, T_in, T_rej),
            "power_output_mw":         self.power_output_w(PLR) / 1e6,
            "heat_input_mw":           self.heat_input_w(PLR, T_in, T_rej) / 1e6,
            "heat_rejection_mw":       self.heat_rejection_w(PLR, T_in, T_rej) / 1e6,
            "eta_carnot":              self.eta_carnot_limit(T_in, T_rej),
            "f_T_reject":              self.f_T_reject(T_rej),
            "f_partload":              self.f_partload(PLR),
            "recuperator_effectiveness": self.recuperator_effectiveness(PLR),
        }
